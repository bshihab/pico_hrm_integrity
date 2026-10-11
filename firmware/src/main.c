#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include "pico/stdlib.h"
#include "pico/multicore.h" 

#include "model_weights.h" 

// --- LED map -----------------------------------------------------------------
// External LEDs on a breadboard, active-high:
//   GPIO -> LED anode (long leg); LED cathode (short leg) -> 330 ohm -> GND.
//   (the resistor may sit on either side of the LED; one per LED, always).
// GP16..GP19 are physical pins 21, 22, 24, 25 (bottom-right corner, away from
// the USB connector); GND is physical pin 23 or 28.
#define LED_NORMAL  16                     // green  : NORMAL beat, 1 blink
#define LED_STYPE   17                     // yellow : S-TYPE beat, 2 blinks
#define LED_VTYPE   18                     // red    : V-TYPE beat, 3 blinks
#define LED_PANIC   19                     // blue   : watchdog, solid then strobe
#define LED_LINK    PICO_DEFAULT_LED_PIN   // onboard GP25: on while the link is armed

static const uint ALL_LEDS[] = {LED_NORMAL, LED_STYPE, LED_VTYPE, LED_PANIC, LED_LINK};
#define N_LEDS (sizeof(ALL_LEDS) / sizeof(ALL_LEDS[0]))

volatile uint32_t last_heartbeat_time = 0;
volatile bool core0_active = false;

#ifndef INPUT_SIZE                 // normally defined by the generated header
#define INPUT_SIZE 10
#define HIDDEN_SIZE 8
#define CLASSES 3
#endif
#ifndef PEAK_THRESHOLD             // trigger constants, normally from the header too
#define PEAK_THRESHOLD 0.6f        // a sample above this followed by a lower one is the R-peak
#define REARM_THRESHOLD 0.3f       // the trigger re-arms once the signal falls below this
#define REFRACTORY_SAMPLES 40      // ignore triggers this soon after the last (320 ms at 125 Hz)
#endif

float history_buffer[INPUT_SIZE]; 
int history_head = 0;

static void leds_init(void) {
    for (uint i = 0; i < N_LEDS; i++) {
        gpio_init(ALL_LEDS[i]);
        gpio_set_dir(ALL_LEDS[i], GPIO_OUT);
        gpio_put(ALL_LEDS[i], 0);
    }
}

static void leds_all(bool on) {
    for (uint i = 0; i < N_LEDS; i++) gpio_put(ALL_LEDS[i], on);
}

static void blink(uint pin, int count, uint32_t on_ms, uint32_t off_ms) {
    for (int i = 0; i < count; i++) {
        gpio_put(pin, 1); sleep_ms(on_ms);
        gpio_put(pin, 0); sleep_ms(off_ms);
    }
}

float relu(float x) { return x > 0 ? x : 0; }

void softmax(float* input, float* output, int len) {
    float sum = 0.0;
    float max_val = input[0];
    for(int i=1; i<len; i++) if(input[i] > max_val) max_val = input[i];
    for(int i=0; i<len; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for(int i=0; i<len; i++) output[i] /= sum;
}

void safety_cop(){
    // 1. Startup: lamp test (every indicator lights once, so a dead LED or a
    //    miswired resistor is caught at boot), then the 3-blink BOOT signal.
    for (uint i = 0; i < N_LEDS; i++) {
        gpio_put(ALL_LEDS[i], 1); sleep_ms(120);
        gpio_put(ALL_LEDS[i], 0);
    }
    blink(LED_LINK, 3, 100, 100);
    
    // 2. Patient Arming
    while (!core0_active) {
        sleep_ms(100);
    }
    gpio_put(LED_LINK, 1);              // link armed: onboard LED stays on
    
    last_heartbeat_time = to_ms_since_boot(get_absolute_time());

    // 3. Watchdog Loop
    while (1){
        uint32_t now = to_ms_since_boot(get_absolute_time());
        
        if (now - last_heartbeat_time > 1000) {
            leds_all(false);
            gpio_put(LED_PANIC, 1);     // solid blue: flatline detected
            sleep_ms(2000);
            
            while(1){                   // hyper-fast blue strobe: system halted
                gpio_put(LED_PANIC, 1); sleep_ms(30);
                gpio_put(LED_PANIC, 0); sleep_ms(30);
            }
        }
        sleep_ms(50);
    }
}

int run_neural_network() {
    float hidden_layer[HIDDEN_SIZE] = {0};
    for (int h = 0; h < HIDDEN_SIZE; h++) { 
        float sum = B1[h]; 
        for (int i = 0; i < INPUT_SIZE; i++) { 
            int idx = (history_head + i) % INPUT_SIZE; 
            sum += history_buffer[idx] * (W1[i][h] * W1_SCALE); 
        }
        hidden_layer[h] = relu(sum);
    }

    float output_logits[CLASSES] = {0}; 
    for (int c = 0; c < CLASSES; c++) {
        output_logits[c] = B2[c]; 
        for (int h = 0; h < HIDDEN_SIZE; h++) { 
            output_logits[c] += hidden_layer[h] * (W2[h][c] * W2_SCALE); 
        }
    }

    float probs[CLASSES];
    softmax(output_logits, probs, CLASSES);

    int winner = 0;
    float max_prob = 0.0;
    for(int c=0; c<CLASSES; c++) {
        if(probs[c] > max_prob) {
            max_prob = probs[c];
            winner = c;
        }
    }
    return winner;
}

void signal_disease(int type) {
    switch (type) {
        case 0: blink(LED_NORMAL, 1, 100,  0); break;   // green,  1 blink
        case 1: blink(LED_STYPE,  2,  80, 80); break;   // yellow, 2 blinks
        case 2: blink(LED_VTYPE,  3,  50, 50); break;   // red,    3 blinks
    }
}

int main(){
    stdio_init_all(); 
    leds_init();
    
    multicore_launch_core1(safety_cop);

    char buffer[64]; 
    int index = 0;
    float last_val = 0.0;   // previous sample, for the peak trigger
    bool armed = true;      // false from a trigger until the signal drops below REARM_THRESHOLD
    int countdown = -1;     // samples still to collect after a peak; -1 = idle
    uint32_t since_trigger = REFRACTORY_SAMPLES;   // samples since the last trigger

    while (1){
        int c = getchar_timeout_us(0); 

        if (c != PICO_ERROR_TIMEOUT){
            if (c == '\n' || c == '}'){
                last_heartbeat_time = to_ms_since_boot(get_absolute_time());
                core0_active = true;

                buffer[index] = '\0'; 
                char *val_ptr = strstr(buffer, "\"val\":");
                if (val_ptr) {
                    float val = atof(val_ptr + 6);   // atof skips the space after the colon

                    history_buffer[history_head] = val;
                    history_head = (history_head + 1) % INPUT_SIZE;

                    // PEAK TRIGGER (the same rule train_tinyml.py uses to cut its windows):
                    // the previous sample was above PEAK_THRESHOLD and this one is lower,
                    // so the previous sample was the R-peak. Triggers are ignored for
                    // REFRACTORY_SAMPLES after the last one (a T wave, never a real beat
                    // below ~190 bpm) and until the signal has dropped below
                    // REARM_THRESHOLD. The model was trained on INPUT_SIZE samples starting
                    // AT the peak, so wait until that many are in the ring buffer (the peak
                    // and this sample are already two of them), then classify once.
                    if (since_trigger < REFRACTORY_SAMPLES) since_trigger++;
                    if (countdown < 0 && armed && since_trigger >= REFRACTORY_SAMPLES
                            && last_val > PEAK_THRESHOLD && val < last_val) {
                        countdown = INPUT_SIZE - 2;
                        armed = false;
                        since_trigger = 0;
                    } else if (countdown > 0) {
                        countdown--;
                    }

                    if (countdown == 0) {
                        countdown = -1;
                        int diagnosis = run_neural_network();
                        const char* names[] = {"NORMAL", "S-TYPE", "V-TYPE"}; 
                        const char* status = names[diagnosis];
                        
                        signal_disease(diagnosis);
                        printf("{\"diagnosis\": \"%s\", \"val\": %.2f}\n", status, val);
                    }

                    if (!armed && val < REARM_THRESHOLD) armed = true;
                    last_val = val;
                }
                index = 0;
            } else { 
                if (index < 63) buffer[index++] = (char)c; 
            }
        }
    }
}
