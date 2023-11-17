cmake_minimum_required(VERSION 3.10)
project(GStream)

# C++
set(CMAKE_CXX_COMPILER g++-7)
set(CMAKE_CXX_FLAGS "-Wall -O3 -lstdc++ -std=c++17 -fopenmp -D_FILE_OFFSET_BITS=64 -DASH_IGNORE_DMESG")

# CUDA
find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER g++-7)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS 
	#-gencode arch=compute_50,code=sm_50
	#-gencode arch=compute_52,code=sm_52
	#-gencode arch=compute_60,code=sm_60
	#-gencode arch=compute_61,code=sm_61
	#-gencode arch=compute_70,code=sm_70
	#-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-O3 -std=c++14 -Xcompiler -fopenmp)

# Boost
set(BOOST_ROOT "~/local")
set(BOOST_INCLUDEDIR "~/local/include")
set(BOOST_LIBRARYDIR "~/local/lib")
find_package(Boost 1.77.0 REQUIRED)

cmake_policy(SET CMP0015 NEW)
include_directories(${CMAKE_SOURCE_DIR}/include ${BOOST_INCLUDEDIR} ${CUDA_INCLUDE_DIRS})
link_directories(${CMAKE_SOURCE_DIR}/lib ${BOOST_LIBRARYDIR})

set(_ASH ash)
set(_GSTREAM gstream)
set(_INFINEL infinel)

# INFINEL
file(GLOB_RECURSE _INFINEL_SRC ${CMAKE_SOURCE_DIR}/source/${_INFINEL}/*)
add_library(${_INFINEL} ${_INFINEL_SRC})
target_link_libraries(${_INFINEL} ${_GSTREAM} ${_ASH})

# TRIANGLE LISTING WITH INFINEL
foreach(_ITEM tl-infinel)
	file(GLOB_RECURSE _SOURCE ${CMAKE_SOURCE_DIR}/source/query/${_ITEM}/*)
	cuda_add_executable(${_ITEM} ${_SOURCE})
	add_dependencies(${_ITEM} ${_INFINEL})
	target_link_libraries(${_ITEM} ${_GSTREAM} ${_ASH} pthread boost_fiber boost_context boost_filesystem boost_system boost_exception)
endforeach()
