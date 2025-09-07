cmake_minimum_required(VERSION 3.10)
project(gemm_benchmark CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(BLAS)

# =================================================================================
# === MODIFICATION: Manual, Foolproof Library Finding                           ===
# =================================================================================
# Step 1: Find the path to the clblast.h header file.
find_path(CLBLAST_INCLUDE_DIR clblast.h
        HINTS /usr/local/include /usr/include)

# Step 2: Find the path to the libclblast.so library file.
find_library(CLBLAST_LIBRARY NAMES clblast
        HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)

# Step 3: Check if both files were found and set HAVE_CLBLAST if they were.
if(CLBLAST_INCLUDE_DIR AND CLBLAST_LIBRARY)
    set(HAVE_CLBLAST TRUE)
    message(STATUS "SUCCESS: Manually found CLBlast library.")
    message(STATUS " -- CLBlast Header: ${CLBLAST_INCLUDE_DIR}")
    message(STATUS " -- CLBlast Library: ${CLBLAST_LIBRARY}")
else()
    set(HAVE_CLBLAST FALSE)
    message(WARNING "CLBlast library or headers not found. The CLBlast test will be skipped.")
endif()
# =================================================================================

add_executable(gemm_benchmark
        main.cpp
        ../lu-decomposition-showcase/extras/utils/opencl_runtime.cpp
        ../lu-decomposition-showcase/extras/utils/init_vector.cpp
)

# --- Preprocessor Definitions ---
if(BLAS_FOUND)
    message(STATUS "Found BLAS library for fast CPU reference: ${BLAS_LIBRARIES}")
    target_compile_definitions(gemm_benchmark PRIVATE HAVE_OPENBLAS)
endif()

if(HAVE_CLBLAST)
    target_compile_definitions(gemm_benchmark PRIVATE HAVE_CLBLAST)
endif()

# --- Include Directories ---
target_include_directories(gemm_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCL_INCLUDE_DIRS}
        ${CLBLAST_INCLUDE_DIR} # <-- Use the variable we just found
)

# Copy kernels folder to the build directory
file(COPY ../lu-decomposition-showcase/extras/kernels DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- Library Linking ---
target_link_libraries(gemm_benchmark PRIVATE
        ${OpenCL_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${CLBLAST_LIBRARY} # <-- Use the variable we just found
)

# Set optimization level and warnings
target_compile_options(gemm_benchmark PRIVATE -O3 -Wall)