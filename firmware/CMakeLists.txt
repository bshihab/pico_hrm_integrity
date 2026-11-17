cmake_minimum_required(VERSION 3.12)

# The Pico SDK location comes from the PICO_SDK_PATH environment variable
# (export PICO_SDK_PATH=/path/to/pico-sdk), as in the SDK's own examples.
# The arm-none-eabi toolchain is found on PATH; if it is somewhere else,
# export PICO_TOOLCHAIN_PATH=/path/to/its/bin.

# --- STANDARD PICO SETUP ---
include(pico_sdk_import.cmake)

project(pico_hrm_integrity C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize SDK
pico_sdk_init()

# Create the Executable from your C file
add_executable(hrm_firmware
    src/main.c
)

# Enable USB Serial (So we can see the output)
pico_enable_stdio_usb(hrm_firmware 1)
pico_enable_stdio_uart(hrm_firmware 0)

# Link Libraries (Multicore, Stdlib)
target_link_libraries(hrm_firmware pico_stdlib pico_multicore)

# Generate the .uf2 file
pico_add_extra_outputs(hrm_firmware)