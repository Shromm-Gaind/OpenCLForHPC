# In 04-async-vectorized/CMakeLists.txt

add_executable(async_vectorized src/main.cpp)

# Make sure to update the filename in your main.cpp to match this
configure_file(src/lu_kernels_optimized.cl ${CMAKE_CURRENT_BINARY_DIR}/lu_kernels_optimized.cl COPYONLY)

# Tell the compiler to aggressively optimize the C++ host code.
target_compile_options(async_vectorized PRIVATE -O3 -Wall -Wextra)

target_link_libraries(async_vectorized PRIVATE common OpenCL::OpenCL)