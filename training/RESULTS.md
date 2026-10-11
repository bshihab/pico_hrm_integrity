# Training results

Dataset `mitbih_test.csv`, seed 42, window 32 samples from the R-peak, jitter ±1, hidden 32, class-weight power 0.5, 27 epochs (early stopping on validation loss).

### Float model, validation split

accuracy 95.0%  macro recall 75.5%

| true \ predicted | NORMAL | S-TYPE | V-TYPE | recall |
|---|---|---|---|---|
| NORMAL | 2649 | 19 | 48 | 97.5% |
| S-TYPE | 39 | 39 | 5 | 47.0% |
| V-TYPE | 39 | 0 | 178 | 82.0% |
| precision | 97.1% | 67.2% | 77.1% | |

### Float model, holdout split (what the Pi streams)

accuracy 94.6%  macro recall 74.8%

| true \ predicted | NORMAL | S-TYPE | V-TYPE | recall |
|---|---|---|---|---|
| NORMAL | 2635 | 7 | 77 | 96.9% |
| S-TYPE | 44 | 35 | 5 | 41.7% |
| V-TYPE | 31 | 0 | 187 | 85.8% |
| precision | 97.2% | 83.3% | 69.5% | |

### Int8 model (as computed by main.c), holdout split

accuracy 94.6%  macro recall 74.8%

| true \ predicted | NORMAL | S-TYPE | V-TYPE | recall |
|---|---|---|---|---|
| NORMAL | 2636 | 7 | 76 | 96.9% |
| S-TYPE | 44 | 35 | 5 | 41.7% |
| V-TYPE | 31 | 0 | 187 | 85.8% |
| precision | 97.2% | 83.3% | 69.8% | |

Int8 vs float agreement on holdout: 99.90%  |  weights 1120 bytes, biases 35 floats

Stream dry run over 3021 holdout beats: 3021 verdicts; 3019 beats gave exactly one verdict, 1 gave none; firmware window identical to the training window for 3019 of 3020 first verdicts.
