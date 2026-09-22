cmake_minimum_required(VERSION 3.14)
project(EdgeAICNNDetection VERSION 1.0 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the source files
set(SOURCES
    src/main.cpp
    src/vision/camera.cpp
    src/vision/classifier.cpp
    src/io_controllers/gpio_controller.cpp
    src/io_controllers/serial_controller.cpp
)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Specify include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find necessary packages
# OpenCV
find_package(OpenCV REQUIRED)

# spdlog
find_package(spdlog REQUIRED)

# Link the libraries to our executable
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    tensorflow-lite
)

# Platform specific linkages (e.g., for Raspberry Pi)
if(UNIX AND NOT APPLE)
    # Link pigpio and pthread for Raspberry Pi hardware control
    target_link_libraries(${PROJECT_NAME} PRIVATE pigpio pthread)
endif()

# Benchmark Executable
set(BENCHMARK_SOURCES
    src/benchmark.cpp
    src/vision/camera.cpp
    src/vision/classifier.cpp
    src/io_controllers/gpio_controller.cpp
    src/io_controllers/serial_controller.cpp
)

add_executable(benchmark_app ${BENCHMARK_SOURCES})

target_include_directories(benchmark_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(benchmark_app PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    tensorflow-lite
)

if(UNIX AND NOT APPLE)
    target_link_libraries(benchmark_app PRIVATE pigpio pthread)
endif()
