# collect all .cu .cpp file into variable IMAGE_PROC_SRCS
file(GLOB_RECURSE SRCS *.cu *.cpp *.hpp)
set(IMAGE_PROC_SRCS ${SRCS} PARENT_SCOPE)