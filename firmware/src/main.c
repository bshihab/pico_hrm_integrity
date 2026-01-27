#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include "pico/stdlib.h"
#include "pico/multicore.h" 

#include "model_weights.h" 

const uint LED_PIN = 25; 

volatile uint32_t last_heartbeat_time = 0;
volatile bool core0_active = false;

#ifndef INPUT_SIZE
#define INPUT_SIZE 10
#define HIDDEN_SIZE 8
#define CLASSES 3
#endif

float history_buffer[INPUT_SIZE]; 
int history_head = 0;

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
    // 1. Startup Signal
    for(int i=0; i<3; i++) {
        gpio_put(LED_PIN, 1); sleep_ms(100);
        gpio_put(LED_PIN, 0); sleep_ms(100);
    }
    
    // 2. Patient Arming
    while (!core0_active) {
        sleep_ms(100);
    }
    
    last_heartbeat_time = to_ms_since_boot(get_absolute_time());

    // 3. Watchdog Loop
    while (1){
        uint32_t now = to_ms_since_boot(get_absolute_time());
        
        if (now - last_heartbeat_time > 1000) {
            gpio_put(LED_PIN, 1);
            sleep_ms(2000);
            
            while(1){ 
                gpio_put(LED_PIN, 1); sleep_ms(30);
                gpio_put(LED_PIN, 0); sleep_ms(30);
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
    if (type == 0) { 
        gpio_put(LED_PIN, 1); sleep_ms(100);
        gpio_put(LED_PIN, 0);
    } else if (type == 1) { 
        for(int i=0; i<2; i++) { gpio_put(LED_PIN, 1); sleep_ms(80); gpio_put(LED_PIN, 0); sleep_ms(80); }
    } else if (type == 2) { 
        for(int i=0; i<3; i++) { gpio_put(LED_PIN, 1); sleep_ms(50); gpio_put(LED_PIN, 0); sleep_ms(50); }
    }
}

int main(){
    stdio_init_all(); 
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    
    multicore_launch_core1(safety_cop);

    char buffer[64]; 
    int index = 0;
    float last_val = 0.0; // Track previous value for peak detection

    while (1){
        int c = getchar_timeout_us(0); 

        if (c != PICO_ERROR_TIMEOUT){
            if (c == '\n' || c == '}'){
                last_heartbeat_time = to_ms_since_boot(get_absolute_time());
                core0_active = true;

                buffer[index] = '\0'; 
                char *val_ptr = strstr(buffer, "\"val\":");
                if (val_ptr) {
                    float val = atof(val_ptr + 7);  

                    history_buffer[history_head] = val;
                    history_head = (history_head + 1) % INPUT_SIZE;

                    // SMART TRIGGER: Peak Detection
                    // Only run if we are high (>0.6) AND we just started dropping (< last_val)
                    // This ensures we catch the TOP of the spike, where the shape is clearest.
                    if (val > 0.6 && val < last_val) {
                        int diagnosis = run_neural_network();
                        const char* names[] = {"NORMAL", "S-TYPE", "V-TYPE"}; 
                        const char* status = names[diagnosis];
                        
                        signal_disease(diagnosis);
                        printf("{\"diagnosis\": \"%s\", \"val\": %.2f}\n", status, val);
                    }
                    
                    last_val = val;
                }
                index = 0;
            } else { 
                if (index < 63) buffer[index++] = (char)c; 
            }
        }
    }
}