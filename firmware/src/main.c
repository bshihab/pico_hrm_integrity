#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"

// config
const uint LED_PIN = 25; // pin for LED on pico

// volatile varibale means that 
volatile uint32_t last_heartbeat_time = 0;
volatile bool core0_active = false;

// CORE 1: checks to see if core 0 is functioning properly

void safety_cop(){
    while (1){
        uint32_t now = to_ms_since_boot(get_absolute_time());

        // if core 0 is active and hasn't updated the time in >1000ms
        if (core0_active && (now - last_heartbeat_time > 1000)) {
            while(1){
                gpio_put(LED_PIN, 1);
                sleep_ms(50);
                gpio_put(LED_PIN, 0);
                sleep_ms(50);
            }
        }
        sleep_ms(100);
    }
}

// CORE 1: The doctor (the one that does the signal processing)

int main(){
    stdio_init_all(); // initialize the USB Serial

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    // wakes up core 1 and tell it to run the safety function
    multicore_launch_core1(safety_cop);

    // Buffer to hold incoming JSON characters
    char buffer[64];
    int index = 0;

    // Variable to track the signal direction
    float last_val = 0.0;

    while (1){
        last_heartbeat_time = to_ms_since_boot(get_absolute_time());
        core0_active = true;


    // Read the data (non-blocking)
    int c = getchar_timeout_us(0); // don't wait for getchar, move on if no data was received

    if (c != PICO_ERROR_TIMEOUT){
        if (c == '\n' || c == '}'){
            buffer[index] = '\0';

            // parse json
            char *val_ptr = strstr(buffer, "\"val\":");
            if (val_ptr) {
                float val = atof(val_ptr + 7); // convert the text to float

                if (val > 0.8 && last_val <= 0.8) {
                    gpio_put(LED_PIN, 1); // turn on LED
                }
                else if (val < 0.6) {
                    gpio_put(LED_PIN, 0); // turn off LED 
                }
                last_val = val;
            }
            index = 0; // reset buffer for next packet
        } else { 
            if (index < 63) {
                buffer[index++] = (char)c;
            }
        }
    }

    }
}