cmake_minimum_required(VERSION 3.12)

# 1. TELL CMAKE TO DOWNLOAD THE SDK IF MISSING
set(PICO_SDK_FETCH_FROM_GIT ON)

# 2. INCLUDE THE SCRIPT YOU JUST MADE
include(pico_sdk_import.cmake)

project(pico_hrm_integrity C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# 3. INITIALIZE THE SDK
pico_sdk_init()

# 4. ADD YOUR C FILE
add_executable(hrm_firmware
    src/main.c
)

# 5. ENABLE USB OUTPUT (So you can see printf)
pico_enable_stdio_usb(hrm_firmware 1)
pico_enable_stdio_uart(hrm_firmware 0)

# 6. LINK LIBRARIES (Multicore support)
target_link_libraries(hrm_firmware pico_stdlib pico_multicore)

# 7. CREATE THE .UF2 FILE (For Drag-and-Drop)
pico_add_extra_outputs(hrm_firmware)