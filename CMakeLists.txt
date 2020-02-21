cmake_minimum_required(VERSION 3.2)

project(TRTNetwork LANGUAGES CXX)

enable_testing()

set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(Boost REQUIRED system iostreams)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/include/)

link_directories(/usr/local/lib/)
link_directories(/usr/local/cuda/lib64/)
link_directories(/usr/local/cuda/lib64/TensorRT/)
link_directories(${PROJECT_SOURCE_DIR}/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse -msse2 -msse3 -mavx -march=native -mfpmath=sse -funroll-loops -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --default-stream per-thread -Wno-deprecated-declarations")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif()

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*")
file(GLOB_RECURSE HEADERS "${PROJECT_SOURCE_DIR}/include/*")

add_library(trt SHARED ${HEADERS} ${SOURCES})
target_link_libraries(trt
    cuda cudart nvToolsExt nvcuvid nvinfer nvparsers
    lzma X11 GL GLEW caffe 
    ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(classification_01 "${PROJECT_SOURCE_DIR}/example/Classification_01.cpp")
target_link_libraries(classification_01 trt)

add_executable(classification_02 "${PROJECT_SOURCE_DIR}/example/Classification_02.cpp")
target_link_libraries(classification_02 trt)

#file(GLOB_RECURSE TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/unit_test/*")
#add_executable(unit_test ${TEST_SOURCES})
#target_link_libraries(unit_test trt)
#add_test(NAME unit_test COMMAND unit_test)

