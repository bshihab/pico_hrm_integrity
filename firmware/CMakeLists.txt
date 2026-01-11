cmake_minimum_required(VERSION 3.12)

# --- MANUAL SDK CONFIGURATION ---
# We point directly to the folder you just cloned.
# This stops VS Code from trying (and failing) to download it.
set(PICO_SDK_PATH "/Users/bilalshihab/dev/pico-sdk")
set(PICO_SDK_FETCH_FROM_GIT OFF)

# Include the helper script
include(pico_sdk_import.cmake)

project(pico_hrm_integrity C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the SDK
pico_sdk_init()

# Add executable
add_executable(hrm_firmware
    src/main.c
)

# Enable USB Output
pico_enable_stdio_usb(hrm_firmware 1)
pico_enable_stdio_uart(hrm_firmware 0)

target_link_libraries(hrm_firmware pico_stdlib pico_multicore)

# Create .uf2 file
pico_add_extra_outputs(hrm_firmware)