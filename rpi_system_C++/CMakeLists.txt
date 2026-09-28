cmake_minimum_required(VERSION 3.14)
project(EdgeAICNNDetection VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")

option(USE_XNNPACK "Enable XNNPACK delegate for TensorFlow Lite" ON)
if(USE_XNNPACK)
  add_definitions(-DUSE_XNNPACK)
endif()

set(TFLITE_PREBUILT_DIR "/home/prathiksabbu/tflite_prebuilt")
set(TFLITE_LIB "${TFLITE_PREBUILT_DIR}/libtensorflow-lite.a")

# Glob every third-party static library XNNPACK/Abseil/ruy/etc. pulled in,
# instead of listing ~115 files by hand.
file(GLOB_RECURSE TFLITE_THIRDPARTY_LIBS_ALL "${TFLITE_PREBUILT_DIR}/libs/*.a")
list(FILTER TFLITE_THIRDPARTY_LIBS_ALL EXCLUDE REGEX "fft4f2d")
set(TFLITE_THIRDPARTY_LIBS ${TFLITE_THIRDPARTY_LIBS_ALL})

set(SOURCES
    src/main.cpp
    src/vision/camera.cpp
    src/vision/classifier.cpp
    src/io_controllers/gpio_controller.cpp
    src/io_controllers/serial_controller.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TFLITE_PREBUILT_DIR}
    ${TFLITE_PREBUILT_DIR}/tflite_thirdparty/flatbuffers/include
)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    -Wl,--start-group
    ${TFLITE_LIB}
    ${TFLITE_THIRDPARTY_LIBS}
    -Wl,--end-group
    pthread
    dl
    m
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE pigpio pthread)
endif()

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
    ${TFLITE_PREBUILT_DIR}
    ${TFLITE_PREBUILT_DIR}/tflite_thirdparty/flatbuffers/include
)

target_link_libraries(benchmark_app PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    -Wl,--start-group
    ${TFLITE_LIB}
    ${TFLITE_THIRDPARTY_LIBS}
    -Wl,--end-group
    pthread
    dl
    m
)

if(UNIX AND NOT APPLE)
    target_link_libraries(benchmark_app PRIVATE pigpio pthread)
endif()