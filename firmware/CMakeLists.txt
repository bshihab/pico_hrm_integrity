cmake_minimum_required(VERSION 3.12)

# --- CONFIGURATION ---
# 1. Force the compiler location (since VS Code is losing it)
set(PICO_TOOLCHAIN_PATH "/usr/local/bin") 

# 2. Point to your manual SDK folder
set(PICO_SDK_PATH "/Users/bilalshihab/dev/pico-sdk")
set(PICO_SDK_FETCH_FROM_GIT OFF)

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

# Link Libraries (Math, Multicore, Stdlib)
target_link_libraries(hrm_firmware pico_stdlib pico_multicore m)

# Generate the .uf2 file
pico_add_extra_outputs(hrm_firmware)