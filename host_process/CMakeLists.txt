# collect all .cu .cpp file into variable HOST_SRCS
file(GLOB_RECURSE SRCS *.cu *.cpp *.hpp)
set(HOST_SRCS ${SRCS} PARENT_SCOPE)