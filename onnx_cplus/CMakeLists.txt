cmake_minimum_required(VERSION 3.28)
# project name
PROJECT(onnx_cplus)
set(CMAKE_CXX_STANDARD 14)

set(ONNXRUNTIME_ROOT_DIR /usr/local)

set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)

# find required opencv
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: 	${OpenCV_VERSION}")
message(STATUS "    headers: 	${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: 	${OpenCV_LIBS}")

# # directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime)
include_directories(include)

# name of executable file and path of source file
file(GLOB My_Source_Files src/*.cpp)
add_executable(onnx_cplus ${My_Source_Files})
# directory of opencv library

# opencv libraries
target_link_libraries(onnx_cplus ${OpenCV_LIBS} ${onnxruntime_lib})


# set(onnxruntime_INSTALL_PREFIX /usr/local)
# set(onnxruntime_INCLUDE_DIRS
#   ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
#   ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/core/session
# )

# include_directories(${onnxruntime_INCLUDE_DIRS})