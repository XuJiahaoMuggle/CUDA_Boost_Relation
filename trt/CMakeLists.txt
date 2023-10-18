# collect all .cu .cpp file into variable TRT_SRCS
file(GLOB_RECURSE SRCS *.cpp *.hpp)
set(TRT_SRCS ${SRCS} PARENT_SCOPE)