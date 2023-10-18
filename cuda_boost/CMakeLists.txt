# collect all .cu .cpp file into variable CUDA_BOOST_SRCS
file(GLOB_RECURSE SRCS *.cu *.cpp *.hpp)
set(CUDA_BOOST_SRCS ${SRCS} PARENT_SCOPE)
