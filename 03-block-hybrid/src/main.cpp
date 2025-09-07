//
// Created by Shromm
//

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cstring>
// Include the OpenCL C header for the C API
#include <CL/cl.h>


#include "helpers.h"
#include "verification.h"

// Let's verify with a manageable size first
#define N 2048 // set to 8192 after to occupy SM depends on your GPU though
#define BLOCK_SIZE 32 // Must match kernel and be a divisor of N


int main() {
    srand(time(NULL));

    if (N % BLOCK_SIZE != 0) {
        std::cerr << "Error: N (" << N << ") must be divisible by BLOCK_SIZE (" << BLOCK_SIZE << ")" << std::endl;
        return 1;
    }

    std::cout << "Setting up host data for a " << N << "x" << N << " matrix..." << std::endl;
    float* h_A_original = new float[N * N];
    initialize_matrix(h_A_original, N);
    float* h_A_working = new float[N * N];
    memcpy(h_A_working, h_A_original, sizeof(float) * N * N);

    std::cout << "\n--- Setting up OpenCL ---" << std::endl;
    cl_int err;

    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    std::cout << "Available Platforms:" << std::endl;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char platform_name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        std::cout << "  " << i << ": " << platform_name << std::endl;
    }
    std::cout << "Select a platform: ";
    unsigned int plat_idx;
    std::cin >> plat_idx;
    cl_platform_id platform = platforms[plat_idx];

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL);
    std::cout << "Available Devices:" << std::endl;
    for (cl_uint i = 0; i < num_devices; ++i) {
        char device_name[128];
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        std::cout << "  " << i << ": " << device_name << std::endl;
    }
    std::cout << "Select a device: ";
    unsigned int dev_idx;
    std::cin >> dev_idx;
    cl_device_id device = devices[dev_idx];

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err); CL_CHECK(err);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N, NULL, &err); CL_CHECK(err);

    float* h_Result = nullptr;
    {
        std::cout << "\n--- Running Block LU Hybrid (CPU+GPU) Implementation (Unoptimized Kernels) ---" << std::endl;
        std::string kernel_source = read_kernel_file("lu_kernels_block.cl");
        const char* source_ptr = kernel_source.c_str();
        cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, NULL, &err); CL_CHECK(err);
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
            std::cerr << "Kernel Build Error: " << log.data() << std::endl;
            exit(1);
        }

        cl_kernel row_update_kernel = clCreateKernel(program, "lu_row_update_kernel", &err); CL_CHECK(err);
        cl_kernel col_update_kernel = clCreateKernel(program, "lu_col_update_kernel", &err); CL_CHECK(err);
        cl_kernel gemm_kernel = clCreateKernel(program, "gemm_update_kernel", &err); CL_CHECK(err);

        err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, sizeof(float) * N * N, h_A_working, 0, NULL, NULL); CL_CHECK(err);

        float* h_panel = new float[BLOCK_SIZE * BLOCK_SIZE];

        int matrix_size = N;
        clSetKernelArg(row_update_kernel, 0, sizeof(cl_mem), &d_A); clSetKernelArg(row_update_kernel, 2, sizeof(int), &matrix_size);
        clSetKernelArg(col_update_kernel, 0, sizeof(cl_mem), &d_A); clSetKernelArg(col_update_kernel, 2, sizeof(int), &matrix_size);
        clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem), &d_A); clSetKernelArg(gemm_kernel, 2, sizeof(int), &matrix_size);

        clFinish(queue);
        auto start = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < N; k += BLOCK_SIZE) {
            // Step 1: Decompose the diagonal panel A_kk on the CPU
            size_t panel_device_origin[3] = { (size_t)k * sizeof(float), (size_t)k, 0 };
            size_t panel_host_origin[3]   = { 0, 0, 0 };
            size_t panel_region[3]        = { (size_t)BLOCK_SIZE * sizeof(float), (size_t)BLOCK_SIZE, 1 };

            err = clEnqueueReadBufferRect(queue, d_A, CL_TRUE,
                                          panel_device_origin, panel_host_origin, panel_region,
                                          (size_t)N * sizeof(float), 0,
                                          (size_t)BLOCK_SIZE * sizeof(float), 0,
                                          h_panel, 0, NULL, NULL);
            CL_CHECK(err);

            cpu_panel_decompose(h_panel, BLOCK_SIZE);

            err = clEnqueueWriteBufferRect(queue, d_A, CL_TRUE,
                                           panel_device_origin, panel_host_origin, panel_region,
                                           (size_t)N * sizeof(float), 0,
                                           (size_t)BLOCK_SIZE * sizeof(float), 0,
                                           h_panel, 0, NULL, NULL);
            CL_CHECK(err);


            // Step 2 & 3: Update the row and column panels on the GPU
            int remaining_size = N - k - BLOCK_SIZE;
            if (remaining_size > 0) {
                clSetKernelArg(row_update_kernel, 1, sizeof(int), &k);
                clSetKernelArg(col_update_kernel, 1, sizeof(int), &k);
                clSetKernelArg(gemm_kernel, 1, sizeof(int), &k);

                size_t global_work_size_trsm[1] = { (size_t)remaining_size };
                size_t local_work_size_trsm[1] = { BLOCK_SIZE };

                size_t gemm_global_work_size[2] = { (size_t)remaining_size, (size_t)remaining_size };
                size_t gemm_local_work_size[2] = { BLOCK_SIZE, BLOCK_SIZE };

                cl_event update_events[2];

                err = clEnqueueNDRangeKernel(queue, row_update_kernel, 1, NULL, global_work_size_trsm, local_work_size_trsm, 0, NULL, &update_events[0]);
                CL_CHECK(err);

                err = clEnqueueNDRangeKernel(queue, col_update_kernel, 1, nullptr, global_work_size_trsm, local_work_size_trsm, 0, NULL, &update_events[1]);
                CL_CHECK(err);

                err = clEnqueueNDRangeKernel(queue, gemm_kernel, 2, NULL, gemm_global_work_size, gemm_local_work_size, 2, update_events, NULL);
                CL_CHECK(err);

                clReleaseEvent(update_events[0]);
                clReleaseEvent(update_events[1]);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_ocl = end - start;

        double flops = (2.0 / 3.0) * pow(N, 3);
        double mflops_ocl = (flops / time_ocl.count()) / 1e6;
        printf("Block LU Hybrid Time: %f seconds\n", time_ocl.count());
        printf("Block LU Hybrid Performance: %.2f MFLOPS\n", mflops_ocl);

        std::cout << "Reading result from GPU for verification..." << std::endl;
        h_Result = new float[N * N];
        err = clEnqueueReadBuffer(queue, d_A, CL_TRUE, 0, sizeof(float) * N * N, h_Result, 0, NULL, NULL);
        CL_CHECK(err);

        delete[] h_panel;
        clReleaseKernel(row_update_kernel);
        clReleaseKernel(col_update_kernel);
        clReleaseKernel(gemm_kernel);
        clReleaseProgram(program);
    }

    verify_lu_decomposition(h_A_original, h_Result, N);

    clReleaseMemObject(d_A);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete[] h_A_original;
    delete[] h_A_working;
    delete[] h_Result;

    return 0;
}
