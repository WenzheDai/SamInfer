cmake_minimum_required(VERSION 3.28)

project(SamTensorrtCplus)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")


set(ONNXRUNTIME_ROOT_DIR /usr/local)
set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)

set(CUDA_DIR /opt/cuda/)
set(TensorRT_DIR /opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/)

# find required opencv
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: 	${OpenCV_VERSION}")
message(STATUS "    headers: 	${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: 	${OpenCV_LIBS}")

link_directories(${CUDA_DIR}/lib64)
link_directories(${CUDA_DIR}/lib)
link_directories(${TensorRT_DIR}/lib)


# # directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime)
include_directories(${CUDA_DIR}/include)
include_directories(${TensorRT_DIR}/include)
include_directories(include)

# name of executable file and path of source file
file(GLOB My_Source_Files src/*.cpp)

add_executable(SamTensorrtCplus ${My_Source_Files})

# opencv libraries
target_link_libraries(SamTensorrtCplus ${OpenCV_LIBS} ${onnxruntime_lib} cudart nvinfer nvonnxparser nvinfer_plugin)
