//
// Created by Shromm
//

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cstring>
#include <CL/cl.h>

#include "helpers.h"
#include "verification.h"


// Use a manageable size for verification
#define N 2048

int main() {
    srand(time(NULL));

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

    std::string kernel_source = read_kernel_file("lu_kernels.cl");
    const char* source_ptr = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, NULL, &err); CL_CHECK(err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        exit(1);
    }

    cl_kernel row_kernel = clCreateKernel(program, "lu_row_kernel", &err); CL_CHECK(err);
    cl_kernel col_kernel = clCreateKernel(program, "lu_column_kernel", &err); CL_CHECK(err);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N, NULL, &err); CL_CHECK(err);

    std::cout << "\n--- Running Naive OpenCL Implementation ---" << std::endl;

    err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, sizeof(float) * N * N, h_A_working, 0, NULL, NULL); CL_CHECK(err);

    int matrix_size = N;
    clSetKernelArg(row_kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(row_kernel, 2, sizeof(int), &matrix_size);
    clSetKernelArg(col_kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(col_kernel, 2, sizeof(int), &matrix_size);

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; k++) {
        cl_event row_event; // Event to track completion of the row kernel

        // Set arguments for the row kernel
        clSetKernelArg(row_kernel, 1, sizeof(int), &k);
        size_t global_work_size_row = N - k;
        if (global_work_size_row > 0) {
            // Enqueue the row kernel and associate it with row_event
            err = clEnqueueNDRangeKernel(queue, row_kernel, 1, NULL, &global_work_size_row, NULL, 0, NULL, &row_event);
            CL_CHECK(err);
        }


        // Set arguments for the column kernel
        clSetKernelArg(col_kernel, 1, sizeof(int), &k);
        size_t global_work_size_col = N - k - 1;

        if (global_work_size_col > 0) {
            // Enqueue the column kernel.
            // The '1, &row_event' part tells this kernel to wait for row_event to be signaled.
            err = clEnqueueNDRangeKernel(queue, col_kernel, 1, NULL, &global_work_size_col, NULL, 1, &row_event, NULL);
            CL_CHECK(err);
        }

        if (global_work_size_row > 0) {
            clReleaseEvent(row_event);
        }
    }

    // Wait for all enqueued commands to finish before stopping the timer.
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ocl = end - start;

    double flops = (2.0 / 3.0) * pow(N, 3);
    double mflops_ocl = (flops / time_ocl.count()) / 1e6;
    printf("Naive OpenCL Time: %f seconds\n", time_ocl.count());
    printf("Naive OpenCL Performance: %.2f MFLOPS\n", mflops_ocl);


    std::cout << "Reading result from GPU for verification..." << std::endl;
    float* h_Result = new float[N * N];
    err = clEnqueueReadBuffer(queue, d_A, CL_TRUE, 0, sizeof(float) * N * N, h_Result, 0, NULL, NULL); CL_CHECK(err);


    if (h_Result) {
        verify_lu_decomposition(h_A_original, h_Result, N);
    }

    clReleaseMemObject(d_A);
    clReleaseKernel(row_kernel);
    clReleaseKernel(col_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete[] h_A_original;
    delete[] h_A_working;
    if (h_Result) delete[] h_Result;

    return 0;
}
