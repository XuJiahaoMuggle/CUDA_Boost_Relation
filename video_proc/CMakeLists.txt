# collect all .cu .cpp file into variable VIDEO_PROC_SRCS
file(GLOB_RECURSE SRCS *.cpp *.cu *.hpp)
set(VIDEO_PROC_SRCS ${SRCS} PARENT_SCOPE)