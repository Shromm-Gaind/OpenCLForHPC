add_executable(naive_opencl src/main.cpp)

# Copy kernel to build dir so the executable can find it
configure_file(src/lu_kernels.cl ${CMAKE_CURRENT_BINARY_DIR}/lu_kernels.cl COPYONLY)

target_link_libraries(naive_opencl PRIVATE common OpenCL::OpenCL)