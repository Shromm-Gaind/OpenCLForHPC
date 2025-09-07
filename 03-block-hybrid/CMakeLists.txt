add_executable(block_hybrid src/main.cpp)

configure_file(src/lu_kernels_block.cl ${CMAKE_CURRENT_BINARY_DIR}/lu_kernels_block.cl COPYONLY)

target_link_libraries(block_hybrid PRIVATE common OpenCL::OpenCL)