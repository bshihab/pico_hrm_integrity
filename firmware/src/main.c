#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // REQUIRED for exp()
#include "pico/stdlib.h"
#include "pico/multicore.h"


// Layer 1 Weights (10x4)
const float W1[10][4] = {
    {-0.4644, 0.4617, 0.5796, -0.6543},
    {0.5395, 0.2282, -0.0021, -0.1559},
    {0.5895, 0.4525, 1.0481, 0.3166},
    {-0.0279, 0.5127, 0.6287, -0.1302},
    {0.1978, 0.0403, -0.1948, 0.6206},
    {-0.3843, 0.4169, 0.4859, 0.1576},
    {-0.3712, 0.1252, 0.5165, 0.4251},
    {0.2553, -0.5404, 0.4784, 0.3168},
    {0.2024, -0.3267, 0.8424, 0.7270},
    {0.4453, -0.0781, 0.2530, -0.4886},
};

// Layer 1 Biases (4)
const float B1[4] = {0.6092, 0.6355, -1.3608, 0.5798};

// Layer 2 Weights (4)
const float W2[4] = {-1.0334, -0.6889, 1.6458, -1.3600};

// Layer 2 Bias (Scalar)
const float B2 = -0.5160;



float relu(float x) {
    return x > 0 ? x : 0;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// takes 10 samples, returns probability of PVC (0.0 to 1.0)
float model_predict(float input[10]) {
    float h[4]; // Hidden layer output

    //  Layer 1 (Dense + ReLU)
    for (int i = 0; i < 4; i++) {
        float sum = B1[i]; 
        for (int j = 0; j < 10; j++) {
            sum += input[j] * W1[j][i]; 
        }
        h[i] = relu(sum);
    }

    //  Layer 2 (Dense + Sigmoid) 
    float final_sum = B2;
    for (int i = 0; i < 4; i++) {
        final_sum += h[i] * W2[i];
    }

    return sigmoid(final_sum);
}



const uint LED_PIN = 25; 
volatile uint32_t last_heartbeat_time = 0;
volatile bool core0_active = false;

// CORE 1: Safety checker
void safety_cop(){
    while (1){
        uint32_t now = to_ms_since_boot(get_absolute_time());
        if (core0_active && (now - last_heartbeat_time > 1000)) {
            while(1){ // Crash detected, blink rapidly
                gpio_put(LED_PIN, 1); sleep_ms(50);
                gpio_put(LED_PIN, 0); sleep_ms(50);
            }
        }
        sleep_ms(100);
    }
}

// CORE 0: Main Application
int main(){
    stdio_init_all(); 
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    multicore_launch_core1(safety_cop);

    char buffer[64];
    int index = 0;

    // Data buffer length = 10
    float input_buffer[10] = {0}; 
    int buffer_pos = 0;    
    int samples_collected = 0;    // Don't predict until we have 10 samples

    while (1){
        last_heartbeat_time = to_ms_since_boot(get_absolute_time());
        core0_active = true;

        int c = getchar_timeout_us(0); 

        if (c != PICO_ERROR_TIMEOUT){
            if (c == '\n' || c == '}'){
                buffer[index] = '\0';
                
                char *val_ptr = strstr(buffer, "\"val\":");
                if (val_ptr) {
                    float val = atof(val_ptr + 7); 

                    
                    // Add new value to rolling buffer
                    input_buffer[buffer_pos] = val;
                    buffer_pos = (buffer_pos + 1) % 10; // Wrap around 0-9
                    samples_collected++;

                    // Only run if we have enough data (at least 10 samples)
                    if (samples_collected >= 10) {
                        
                        float ordered_input[10];
                        for(int i=0; i<10; i++) {
                            ordered_input[i] = input_buffer[(buffer_pos + i) % 10];
                        }

                        // 3. run the model
                        float prediction = model_predict(ordered_input);

                        // If prediction > 0.5 (50%), it's a PVC
                        if (prediction > 0.5) {
                            printf("PVC DETECTED! Score: %.2f\n", prediction);
                            gpio_put(LED_PIN, 1);
                        } else {
                            gpio_put(LED_PIN, 0); 
                        }
                    }
                }
                index = 0; 
            } else { 
                if (index < 63) buffer[index++] = (char)c;
            }
        }
    }
}