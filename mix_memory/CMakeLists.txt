# collect all .cu .cpp file into variable MEMORY_SRCS
file(GLOB_RECURSE SRCS *.hpp *.cpp)
set(MEMORY_SRCS ${SRCS} PARENT_SCOPE)