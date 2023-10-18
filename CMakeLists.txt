cmake_minimum_required(VERSION 3.16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/workspace)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# prepare cuda 
set(CUDA_DIR "D:/cuda12_relation")
set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 61)
# enable cuda c cxx
project(tiny_cv_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# to link and use cuda
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib/x64)

# prepare tensor-rt
set(TRT_DIR "D:/cuda12_relation/TensorRT-8.6.1.6")
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)

# config opencv
set(OpenCV_DIR "D:/Embedded_Software/opencv/build")
find_package(OpenCV)
if (OpenCV_FOUND)
    message (NOTICE "OpenCV FONUDED!")
else(OpenCV_FOUND)
    message (FATAL_ERROR "OpenCV NOT FONUDED!")
endif(OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})

# config ffmpeg
set(FFmpeg_DIR "D:/big_is_coming/FFmpeg")
include_directories(${FFmpeg_DIR}/include)
link_directories(${FFmpeg_DIR}/lib)

# config CUVID
set(NVCODEC_DIR "D:/Embedded_Software/Video_Codec_SDK_10.0.26")
include_directories(${NVCODEC_DIR}/Interface)
link_directories(${NVCODEC_DIR}/Lib/x64)

# prepare sources
set(MEMORY_SRCS)
add_subdirectory(mix_memory)

set(UTILS_SRCS)
add_subdirectory(utils)

set(CUDA_BOOST_SRCS)
add_subdirectory(cuda_boost)

set(VIDEO_PROC_SRCS)
add_subdirectory(video_proc)

set(IMAGE_PROC_SRCS)
add_subdirectory(image_proc)

set(TRT_SRCS)
add_subdirectory(trt)

set(YOLO_SRCS)
add_subdirectory(yolo)

set(CPM_SRCS)
add_subdirectory(cpm)

set(HOST_SRCS)
add_subdirectory(host_process)

# exe srcs
set(TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# add_executable(benchmark
#     ${TEST_SOURCE_DIR}/benchmark.cpp 
#     ${MEMORY_SRCS} 
#     ${UTILS_SRCS}
#     ${CUDA_BOOST_SRCS}
#     ${VIDEO_PROC_SRCS}
#     ${IMAGE_PROC_SRCS}
#     ${TRT_SRCS}
#     ${TRT_PLUGIN_SRCS}
#     ${HOST_SRCS}
#     ${YOLO_SRCS}
# )

# target_link_libraries(benchmark
#     PRIVATE cudart           # cuda runtime
#     PRIVATE cuda             # cuda
#     PRIVATE ${OpenCV_LIBS}   # opencv
#     PRIVATE nvinfer          # nvinfer
#     PRIVATE nvonnxparser     # nvonnxparser
#     PRIVATE nvinfer_plugin   # nvinfer_plugin
#     PRIVATE avcodec          # avcodec
#     PRIVATE avformat         # avformat
#     PRIVATE avutil           # avutil
#     PRIVATE avdevice         # avdevice
#     PRIVATE nvcuvid
# )

# add_executable(infer_image
#     ${TEST_SOURCE_DIR}/infer_image.cpp 
#     ${MEMORY_SRCS} 
#     ${UTILS_SRCS}
#     ${CUDA_BOOST_SRCS}
#     ${VIDEO_PROC_SRCS}
#     ${IMAGE_PROC_SRCS}
#     ${TRT_SRCS}
#     ${TRT_PLUGIN_SRCS}
#     ${YOLO_SRCS}
# )

# target_link_libraries(infer_image
#     PRIVATE cudart           # cuda runtime
#     PRIVATE cuda             # cuda
#     PRIVATE ${OpenCV_LIBS}   # opencv
#     PRIVATE nvinfer          # nvinfer
#     PRIVATE nvonnxparser     # nvonnxparser
#     PRIVATE nvinfer_plugin   # nvinfer_plugin
#     PRIVATE avcodec          # avcodec
#     PRIVATE avformat         # avformat
#     PRIVATE avutil           # avutil
#     PRIVATE avdevice         # avdevice
#     PRIVATE nvcuvid
# )

add_executable(infer_video
    ${TEST_SOURCE_DIR}/infer_video.cpp 
    ${MEMORY_SRCS} 
    ${UTILS_SRCS}
    ${CUDA_BOOST_SRCS}
    ${VIDEO_PROC_SRCS}
    ${IMAGE_PROC_SRCS}
    ${TRT_SRCS}
    ${TRT_PLUGIN_SRCS}
    ${YOLO_SRCS}
)

target_link_libraries(infer_video
    PRIVATE cudart           # cuda runtime
    PRIVATE cuda             # cuda
    PRIVATE ${OpenCV_LIBS}   # opencv
    PRIVATE nvinfer          # nvinfer
    PRIVATE nvonnxparser     # nvonnxparser
    PRIVATE nvinfer_plugin   # nvinfer_plugin
    PRIVATE avcodec          # avcodec
    PRIVATE avformat         # avformat
    PRIVATE avutil           # avutil
    PRIVATE avdevice         # avdevice
    PRIVATE nvcuvid
)

# add_executable(infer_cpm
#     ${TEST_SOURCE_DIR}/infer_cpm.cpp 
#     ${MEMORY_SRCS} 
#     ${UTILS_SRCS}
#     ${CUDA_BOOST_SRCS}
#     ${VIDEO_PROC_SRCS}
#     ${IMAGE_PROC_SRCS}
#     ${TRT_SRCS}
#     ${TRT_PLUGIN_SRCS}
#     ${YOLO_SRCS}
# )

# target_link_libraries(infer_cpm
#     PRIVATE cudart           # cuda runtime
#     PRIVATE cuda             # cuda
#     PRIVATE ${OpenCV_LIBS}   # opencv
#     PRIVATE nvinfer          # nvinfer
#     PRIVATE nvonnxparser     # nvonnxparser
#     PRIVATE nvinfer_plugin   # nvinfer_plugin
#     PRIVATE avcodec          # avcodec
#     PRIVATE avformat         # avformat
#     PRIVATE avutil           # avutil
#     PRIVATE avdevice         # avdevice
# )

# add_executable(infer_hardware
#     ${TEST_SOURCE_DIR}/infer_hardware.cpp 
#     ${MEMORY_SRCS} 
#     ${UTILS_SRCS}
#     ${CUDA_BOOST_SRCS}
#     ${VIDEO_PROC_SRCS}
#     ${IMAGE_PROC_SRCS}
#     ${TRT_SRCS}
#     ${TRT_PLUGIN_SRCS}
#     ${YOLO_SRCS}
# )

# target_link_libraries(infer_hardware
#     PRIVATE cudart           # cuda runtime
#     PRIVATE cuda             # cuda
#     PRIVATE ${OpenCV_LIBS}   # opencv
#     PRIVATE nvinfer          # nvinfer
#     PRIVATE nvonnxparser     # nvonnxparser
#     PRIVATE nvinfer_plugin   # nvinfer_plugin
#     PRIVATE avcodec          # avcodec
#     PRIVATE avformat         # avformat
#     PRIVATE avutil           # avutil
#     PRIVATE avdevice         # avdevice
#     PRIVATE nvcuvid
# )
