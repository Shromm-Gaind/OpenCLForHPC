cmake_minimum_required(VERSION 3.15)
project(LUDecompositionShowcase LANGUAGES C CXX)

# Set modern C/C++ standards for the whole project
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL once for all subprojects
find_package(OpenCL REQUIRED)
message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")

# Add our common helper library first
add_subdirectory(common)

# Add each optimization stage
add_subdirectory(01-sequential-cpu)
add_subdirectory(02-naive-opencl)
add_subdirectory(03-block-hybrid)
add_subdirectory(04-async-vectorized)

# Note: The GEMM benchmark in 'extras' is not part of the main build.
# See the README for instructions on how to build it separately.