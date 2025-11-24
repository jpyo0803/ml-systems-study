cmake_minimum_required(VERSION 3.10)
project(mnist_onnxruntime)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime 위치 (직접 path 맞춰야 함)
set(ONNXRUNTIME_DIR "/opt/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(server server.cpp)

target_link_libraries(server onnxruntime)