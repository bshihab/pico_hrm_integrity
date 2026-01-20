#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include "pico/stdlib.h"
#include "pico/multicore.h" // included the multicore library so that we use one core as the safety watch and the other to run the inference of the ML model

// INCLUDE THE GENERATED WEIGHTS FILE
// This replaces the hardcoded W1, B1, etc.
#include "model_weights.h"

// config
const uint LED_PIN = 25; 

// used volatile variable so that both cores can use the variables and will not be optimized in CPU reg since it's in RAM
volatile uint32_t last_heartbeat_time = 0;
volatile bool core0_active = false;

// If constants aren't in the header, define defaults here
#ifndef INPUT_SIZE
#define INPUT_SIZE 10
#define HIDDEN_SIZE 8
#define CLASSES 3
#endif

float history_buffer[INPUT_SIZE]; // storing 10 individual readings at a time, which is what the model uses to check pulse
int history_head = 0;

// math helpers

// relu is used because we only want neurons to fire when there is a positive correlation, not negative.  
float relu(float x) { return x > 0 ? x : 0; }

// converts raw numbers to values between 0 and 1 to act as probabilities for the weights
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

//  CORE 1: Safety Watcher
void safety_cop(){
    while (1){
        uint32_t now = to_ms_since_boot(get_absolute_time());
        if (core0_active && (now - last_heartbeat_time > 1000)) {
            while(1){ // purpose of this is to continuously toggle the LEDs, we've determined that core0 is dead
                gpio_put(LED_PIN, 1); sleep_ms(25);
                gpio_put(LED_PIN, 0); sleep_ms(25);
            }
        }
        sleep_ms(100);
    }
}

// INFERENCE
int run_neural_network() {
    float hidden_layer[HIDDEN_SIZE] = {0};
    
    // Layer 1: Input -> Hidden, apply the multiplier
    for (int h = 0; h < HIDDEN_SIZE; h++) { // h represents each NEURON in layer 
        float sum = B1[h]; // start accumulating bias for this neuron
        for (int i = 0; i < INPUT_SIZE; i++) { // iterates thru input features of current neuron
            int idx = (history_head + i) % INPUT_SIZE; // circular buffer 
            // De-quantize: weight * scale
            sum += history_buffer[idx] * (W1[i][h] * W1_SCALE); 
        }
        hidden_layer[h] = relu(sum);
    }

    // Layer 2: Hidden -> Output Logits (Apply W2_SCALE here!)
    float output_logits[CLASSES] = {0}; // output number depending on how many classes we have
    for (int c = 0; c < CLASSES; c++) {
        output_logits[c] = B2[c]; // for every neuron that we have, we start with output bias
        for (int h = 0; h < HIDDEN_SIZE; h++) { 
            output_logits[c] += hidden_layer[h] * (W2[h][c] * W2_SCALE); // calculate score for the final class 
        }
    }

    // Softmax: Logits -> Probabilities
    // normalize outputs to probabilites
    float probs[CLASSES];
    softmax(output_logits, probs, CLASSES);

    // which output won?
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

// LED patterns for different disease classifications
void signal_disease(int type) {
    if (type == 0) { // normal: 1 Short Blink
        gpio_put(LED_PIN, 1); sleep_ms(100);
        gpio_put(LED_PIN, 0);
    } else if (type == 1) { // s-type: 2 Fast Blinks
        for(int i=0; i<2; i++) { gpio_put(LED_PIN, 1); sleep_ms(80); gpio_put(LED_PIN, 0); sleep_ms(80); }
    } else if (type == 2) { // v-type: 3 Fast Blinks
        for(int i=0; i<3; i++) { gpio_put(LED_PIN, 1); sleep_ms(50); gpio_put(LED_PIN, 0); sleep_ms(50); }
    }
}

// core 0
int main(){
    stdio_init_all(); 
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    multicore_launch_core1(safety_cop);

    char buffer[64]; // 64 byte buffer allocated for json packet
    int index = 0;

    while (1){
        last_heartbeat_time = to_ms_since_boot(get_absolute_time());
        core0_active = true;

        int c = getchar_timeout_us(0); // non-blocking, polling. if empty, return to start of loop immediately to not waste time

        if (c != PICO_ERROR_TIMEOUT){
            if (c == '\n' || c == '}'){
                buffer[index] = '\0'; // if we've collected the entire data packet, add a null terminator to the string
                char *val_ptr = strstr(buffer, "\"val\":");
                if (val_ptr) {
                    float val = atof(val_ptr + 7);  

                    // update the buffer
                    history_buffer[history_head] = val;
                    history_head = (history_head + 1) % INPUT_SIZE;

                    // Trigger only on peaks to save CPU
                    if (val > 0.6) {
                        int diagnosis = run_neural_network();
                        
                        const char* names[] = {"NORMAL", "S-TYPE", "V-TYPE"}; 
                        const char* status = names[diagnosis];
                        
                        signal_disease(diagnosis);

                        printf("{\"diagnosis\": \"%s\", \"val\": %.2f}\n", status, val);
                    }
                }
                index = 0;
            } else { 
                if (index < 63) buffer[index++] = (char)c; // prevent buffer overflow 
            }
        }
    }
}