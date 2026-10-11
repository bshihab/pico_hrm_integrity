"""
Train the TinyML heartbeat classifier and emit firmware/src/model_weights.h.

The window the model sees is defined ONCE, here, and mirrored by the firmware:

  The firmware watches the sample stream. When the previous sample was above
  PEAK_THRESHOLD and the current one is lower, the previous sample was the
  R-peak. A trigger is ignored for REFRACTORY_SAMPLES after the last one
  (rejects T waves, never a real beat below ~190 bpm). The firmware then waits
  until INPUT_SIZE samples, starting at the peak, sit in its ring buffer and
  classifies them once.

  This script finds the peak in each dataset row with the same rule and cuts
  the same INPUT_SIZE-sample window, so training and inference see identical
  slices ("dynamic peak alignment"). Training windows are also cut one sample
  early and one sample late (jitter) so a trigger that is off by one sample on
  the device still lands inside what the model has seen.

Data handling:
  * rows are shuffled with a fixed seed and split per class (stratified) into
    train / validation / holdout, because the Kaggle CSV is sorted by label;
  * classes are weighted inversely to their frequency (normal beats outnumber
    S beats about 30:1);
  * the holdout rows are written to hospital_sim/holdout.json so the Pi only
    ever streams beats the model has never seen. Each entry also carries how
    many samples of the row to stream: dataset rows are 1.2 heart periods
    long, so they contain the onset of the NEXT beat, and the simulator cuts
    the row just before it so one row equals one beat on the wire;
  * the int8 export is re-run in numpy exactly as the C code computes it and
    compared against the float model.

Usage:
  python3 training/train_tinyml.py                 # finds or downloads the CSV
  python3 training/train_tinyml.py --csv /path/to/mitbih_test.csv
"""
import argparse
import json
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# --- model / window configuration (also written into the C header) ----------
INPUT_SIZE = 32          # samples per window, starting at the R-peak (256 ms at 125 Hz)
HIDDEN_SIZE = 32
CLASSES = 3
PEAK_THRESHOLD = 0.6     # firmware trigger level (normalised units)
REARM_THRESHOLD = 0.3    # firmware re-arms the trigger once the signal falls below this
REFRACTORY_SAMPLES = 40  # firmware ignores triggers this soon after the last one (320 ms)
INTER_BEAT_GAP = 8       # zero samples the simulator streams after each beat (read from holdout.json)
# Every streamed beat carries at least INPUT_SIZE samples after its peak plus the
# gap, so the next beat's peak always falls outside the refractory period.
assert INPUT_SIZE + INTER_BEAT_GAP >= REFRACTORY_SAMPLES
JITTER = 1               # train on windows shifted by -1, 0, +1 samples
CLASS_WEIGHT_POWER = 0.5 # 1.0 = fully balanced classes (max V recall, many false reds);
                         # 0.5 chosen on the validation split: best macro-F1, 98% of normals stay green
SEED = 42
SPLIT = (0.70, 0.15, 0.15)   # train / validation / holdout, per class
CLASS_NAMES = ["NORMAL", "S-TYPE", "V-TYPE"]

REPO = Path(__file__).resolve().parent.parent
HEADER_PATH = REPO / "firmware" / "src" / "model_weights.h"
HOLDOUT_PATH = REPO / "hospital_sim" / "holdout.json"
RESULTS_PATH = REPO / "training" / "RESULTS.md"
CSV_FILENAME = "mitbih_test.csv"

# --- Google Cloud Storage (used only when no local CSV is found) -------------
GCP_API_KEY = os.getenv("GCP_API_KEY")
GCP_BUCKET_NAME = "heart-data-repo-1"
GCP_OBJECT_NAME = "mitbih_test.csv"


def get_google_cloud_url():
    encoded = urllib.parse.quote(GCP_OBJECT_NAME, safe="")
    return (f"https://storage.googleapis.com/storage/v1/b/{GCP_BUCKET_NAME}"
            f"/o/{encoded}?alt=media&key={GCP_API_KEY}")


def locate_csv(explicit):
    """Prefer an explicit path, then known local copies, then download."""
    candidates = [explicit] if explicit else []
    candidates += [REPO / "training" / CSV_FILENAME, REPO / "hospital_sim" / CSV_FILENAME]
    for c in candidates:
        if c and Path(c).exists() and Path(c).stat().st_size > 1000:
            return Path(c)
    if not GCP_API_KEY:
        raise SystemExit("No local CSV and no GCP_API_KEY in .env; cannot download.")
    target = REPO / "training" / CSV_FILENAME
    print("Downloading dataset from Google Cloud Storage ...")
    r = requests.get(get_google_cloud_url(), stream=True)
    if r.status_code != 200:
        raise SystemExit(f"Download failed: HTTP {r.status_code}: {r.text[:200]}")
    with open(target, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return target


# --- window extraction (the rule shared with main.c) --------------------------
def triggers(row, thr=PEAK_THRESHOLD, rearm=REARM_THRESHOLD, refractory=REFRACTORY_SAMPLES):
    """Peak indices the firmware would act on while this row streams past it.

    Exactly mirrors main.c: fire when the previous sample was above `thr` and
    the current one is lower (the previous sample is the peak), stay disarmed
    until the signal has dropped below `rearm`, ignore triggers closer than
    `refractory` samples to the previous one. The sample before the row is
    taken as 0, which is what the simulator's inter-beat baseline looks like.
    """
    out, last, armed, last_trig = [], 0.0, True, -10 ** 9
    for i, v in enumerate(row):
        if armed and last > thr and v < last and (i - last_trig) >= refractory:
            out.append(i - 1)
            armed, last_trig = False, i
        if not armed and v < rearm:
            armed = True
        last = v
    return out


def find_peak(row):
    """Index of the first R-peak as the firmware would find it, or -1."""
    t = triggers(row)
    return t[0] if t else -1


def stream_length(row):
    """How many samples of the row the simulator should send.

    Up to just before the next beat's R-peak if the row contains one, else up
    to the last non-zero sample (the rest is the dataset's zero padding), and
    never fewer than INPUT_SIZE samples after the peak so the whole window the
    model sees comes from this row.
    """
    t = triggers(row)
    if not t:
        return len(row)
    floor = min(t[0] + INPUT_SIZE, len(row))
    if len(t) > 1:
        cut = t[1] - 1                      # index of the next beat's peak
    else:
        nz = np.flatnonzero(row)
        cut = int(nz[-1]) + 1 if nz.size else len(row)
    # Never end on the next beat's upstroke: the last streamed sample must be
    # at or below PEAK_THRESHOLD, otherwise the drop to the baseline gap would
    # itself look like a peak to the firmware.
    while cut > floor and row[cut - 1] > PEAK_THRESHOLD:
        cut -= 1
    return int(max(cut, floor))


def stream_selfcheck(rows):
    """Dry run of the HIL loop in Python: stream the beats exactly as the
    simulator will (trimmed rows, baseline gap) through a line-for-line copy of
    main.c's trigger state machine, and compare every window it would classify
    with the window this script cut for that beat. Returns a short report."""
    N = INPUT_SIZE
    buf = np.zeros(N, np.float32); head = 0; last = 0.0; armed = True
    countdown = -1; since = REFRACTORY_SAMPLES
    verdicts = {}
    gap = np.zeros(INTER_BEAT_GAP, np.float32)
    for r, row in enumerate(rows):
        for val in np.concatenate([row[:stream_length(row)], gap]):
            buf[head] = val; head = (head + 1) % N
            if since < REFRACTORY_SAMPLES: since += 1
            if countdown < 0 and armed and since >= REFRACTORY_SAMPLES and last > PEAK_THRESHOLD and val < last:
                countdown, armed, since = N - 2, False, 0
            elif countdown > 0:
                countdown -= 1
            if countdown == 0:
                countdown = -1
                verdicts.setdefault(r, []).append(np.array([buf[(head + i) % N] for i in range(N)]))
            if not armed and val < REARM_THRESHOLD:
                armed = True
            last = val
    one = sum(1 for v in verdicts.values() if len(v) == 1)
    match = sum(1 for r, v in verdicts.items()
                if find_peak(rows[r]) >= 0 and np.allclose(v[0], cut_window(rows[r], find_peak(rows[r]))))
    return (f"Stream dry run over {len(rows)} holdout beats: {sum(map(len, verdicts.values()))} verdicts; "
            f"{one} beats gave exactly one verdict, {len(rows) - len(verdicts)} gave none; "
            f"firmware window identical to the training window for {match} of {len(verdicts)} first verdicts.")


def cut_window(row, start):
    """INPUT_SIZE samples from `start`, zero-padded like the dataset itself."""
    w = row[max(start, 0): start + INPUT_SIZE]
    if start < 0:
        w = np.concatenate([np.zeros(-start, dtype=row.dtype), w])
    if w.size < INPUT_SIZE:
        w = np.concatenate([w, np.zeros(INPUT_SIZE - w.size, dtype=row.dtype)])
    return w


def windows_for(rows, labels, shifts):
    X, y, skipped = [], [], 0
    for row, lab in zip(rows, labels):
        p = find_peak(row)
        if p < 0:
            skipped += 1
            continue
        for s in shifts:
            X.append(cut_window(row, p + s))
            y.append(lab)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), skipped


# --- data ---------------------------------------------------------------------
def load_and_split(csv_path, seed):
    df = pd.read_csv(csv_path, header=None)
    df = df[df[187].isin([0.0, 1.0, 2.0])]
    rows = df.iloc[:, :187].values.astype(np.float32)
    labels = df[187].values.astype(np.int64)
    raw_index = df.index.values                     # row numbers in the CSV file

    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "holdout": []}
    for cls in range(CLASSES):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        a = int(n * SPLIT[0])
        b = a + int(n * SPLIT[1])
        parts["train"].append(idx[:a])
        parts["val"].append(idx[a:b])
        parts["holdout"].append(idx[b:])
    parts = {k: np.sort(np.concatenate(v)) for k, v in parts.items()}
    return rows, labels, raw_index, parts


# --- metrics ------------------------------------------------------------------
def confusion(y_true, y_pred):
    m = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def summarise(name, y_true, y_pred):
    m = confusion(y_true, y_pred)
    acc = np.trace(m) / m.sum()
    recall = np.diag(m) / np.maximum(m.sum(axis=1), 1)
    precision = np.diag(m) / np.maximum(m.sum(axis=0), 1)
    lines = [f"### {name}", "",
             f"accuracy {acc*100:.1f}%  macro recall {recall.mean()*100:.1f}%", "",
             "| true \\ predicted | " + " | ".join(CLASS_NAMES) + " | recall |",
             "|---|" + "---|" * (CLASSES + 1)]
    for i, cn in enumerate(CLASS_NAMES):
        lines.append(f"| {cn} | " + " | ".join(str(v) for v in m[i]) + f" | {recall[i]*100:.1f}% |")
    lines.append("| precision | " + " | ".join(f"{p*100:.1f}%" for p in precision) + " | |")
    text = "\n".join(lines)
    print(text, "\n")
    return text, acc, recall


# --- quantisation + C export --------------------------------------------------
def quantise(w):
    scale = float(np.max(np.abs(w)) / 127.0) or 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def int8_forward(X, W1q, s1, b1, W2q, s2, b2):
    """Exactly what run_neural_network() in main.c computes."""
    h = np.maximum(X @ (W1q.astype(np.float32) * np.float32(s1)) + b1, 0)
    logits = h @ (W2q.astype(np.float32) * np.float32(s2)) + b2
    return logits.argmax(axis=1)


def write_header(W1q, s1, b1, W2q, s2, b2):
    def mat(name, q):
        body = ",\n".join("    {" + ", ".join(f"{v:4d}" for v in row) + "}" for row in q)
        return f"const int8_t {name}[{q.shape[0]}][{q.shape[1]}] = {{\n{body},\n}};\n"

    def vec(name, b):
        return f"const float {name}[{b.size}] = {{ " + ", ".join(f"{v:.6f}" for v in b) + " };\n"

    c = ["// Auto-generated by training/train_tinyml.py. Do not edit by hand.",
         "#ifndef MODEL_WEIGHTS_H", "#define MODEL_WEIGHTS_H", "", "#include <stdint.h>", "",
         "// Trigger and window definition shared with main.c (see train_tinyml.py):",
         "// the R-peak is a sample above PEAK_THRESHOLD followed by a lower sample;",
         "// triggers within REFRACTORY_SAMPLES of the last are ignored; the model sees",
         "// INPUT_SIZE samples starting at the peak.",
         f"#define INPUT_SIZE {INPUT_SIZE}", f"#define HIDDEN_SIZE {HIDDEN_SIZE}",
         f"#define CLASSES {CLASSES}", f"#define PEAK_THRESHOLD {PEAK_THRESHOLD}f",
         f"#define REARM_THRESHOLD {REARM_THRESHOLD}f", f"#define REFRACTORY_SAMPLES {REFRACTORY_SAMPLES}", "",
         "// Layer 1 (hidden): int8 weights, per-tensor symmetric scale, float bias",
         f"const float W1_SCALE = {s1:.12f};", mat("W1", W1q), vec("B1", b1),
         "// Layer 2 (output)",
         f"const float W2_SCALE = {s2:.12f};", mat("W2", W2q), vec("B2", b2),
         "#endif // MODEL_WEIGHTS_H", ""]
    HEADER_PATH.write_text("\n".join(c))
    print(f"Header written: {HEADER_PATH}")


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="path to mitbih_test.csv (default: look locally, else download)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    tf.keras.utils.set_random_seed(args.seed)

    csv_path = locate_csv(args.csv)
    print(f"Dataset: {csv_path}")
    rows, labels, raw_index, parts = load_and_split(csv_path, args.seed)
    for k, idx in parts.items():
        counts = np.bincount(labels[idx], minlength=CLASSES)
        print(f"{k:8s} rows {len(idx):6d}  N {counts[0]:5d}  S {counts[1]:4d}  V {counts[2]:4d}")

    shifts = list(range(-JITTER, JITTER + 1))
    Xtr, ytr, sk_tr = windows_for(rows[parts["train"]], labels[parts["train"]], shifts)
    Xva, yva, sk_va = windows_for(rows[parts["val"]], labels[parts["val"]], [0])
    Xho, yho, sk_ho = windows_for(rows[parts["holdout"]], labels[parts["holdout"]], [0])
    print(f"windows: train {len(Xtr)} (x{len(shifts)} jitter)  val {len(Xva)}  holdout {len(Xho)}"
          f"  | rows without a firmware-style peak: {sk_tr + sk_va + sk_ho}")

    counts = np.bincount(ytr, minlength=CLASSES).astype(np.float64)
    class_weight = {c: float((len(ytr) / (CLASSES * counts[c])) ** CLASS_WEIGHT_POWER) for c in range(CLASSES)}
    print("class weights:", {CLASS_NAMES[c]: round(w, 2) for c, w in class_weight.items()})

    model = Sequential([Input((INPUT_SIZE,)),
                        Dense(HIDDEN_SIZE, activation="relu"),
                        Dense(CLASSES, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=64,
                     class_weight=class_weight, callbacks=[early], verbose=0)
    print(f"trained {len(hist.history['loss'])} epochs (early stopping on val_loss)")

    report = [f"# Training results", "",
              f"Dataset `{csv_path.name}`, seed {args.seed}, window {INPUT_SIZE} samples from the R-peak, "
              f"jitter ±{JITTER}, hidden {HIDDEN_SIZE}, class-weight power {CLASS_WEIGHT_POWER}, "
              f"{len(hist.history['loss'])} epochs (early stopping on validation loss).", ""]

    pva = model.predict(Xva, verbose=0).argmax(axis=1)
    pho = model.predict(Xho, verbose=0).argmax(axis=1)
    t, _, _ = summarise("Float model, validation split", yva, pva); report += [t, ""]
    t, acc_f, rec_f = summarise("Float model, holdout split (what the Pi streams)", yho, pho); report += [t, ""]

    dense = [l for l in model.layers if isinstance(l, Dense)]
    W1, b1 = dense[0].get_weights()
    W2, b2 = dense[1].get_weights()
    W1q, s1 = quantise(W1)
    W2q, s2 = quantise(W2)
    pq = int8_forward(Xho, W1q, s1, b1, W2q, s2, b2)
    t, acc_q, rec_q = summarise("Int8 model (as computed by main.c), holdout split", yho, pq); report += [t, ""]
    agree = float((pq == pho).mean())
    line = f"Int8 vs float agreement on holdout: {agree*100:.2f}%  |  weights {W1q.size + W2q.size} bytes, biases {b1.size + b2.size} floats"
    print(line); report += [line, ""]

    line = stream_selfcheck(rows[parts["holdout"]])
    print(line); report += [line, ""]

    write_header(W1q, s1, b1, W2q, s2, b2)
    HOLDOUT_PATH.write_text(json.dumps({
        "csv": csv_path.name, "seed": args.seed, "input_size": INPUT_SIZE,
        "inter_beat_gap": INTER_BEAT_GAP,
        "note": "[csv_row, samples_to_stream] for rows never used in training or validation; "
                "hospital_monitor.py streams only these, cut before the next beat's R-peak.",
        "rows": [[int(raw_index[i]), int(stream_length(rows[i]))] for i in parts["holdout"]]},
        separators=(",", ":")))
    print(f"Holdout list written: {HOLDOUT_PATH} ({len(parts['holdout'])} rows)")
    RESULTS_PATH.write_text("\n".join(report))
    print(f"Report written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
