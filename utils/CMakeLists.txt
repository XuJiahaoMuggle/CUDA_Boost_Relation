# collect all .cu .cpp file into variable UTILS_SRCS
file(GLOB_RECURSE SRCS *.hpp *.cpp)
set(UTILS_SRCS ${SRCS} PARENT_SCOPE)