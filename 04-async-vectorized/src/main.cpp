//
// Created by Shromm
//
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

// Include the OpenCL C header for the C API
#include <CL/cl.h>
#include <cstring>


#include "helpers.h"

// matrix size set to 8192 to cause all the SM's to be active
#define N 2048
#define BLOCK_SIZE 32 // Must match kernel and be a divisor of N
#define VEC_SIZE 16
#define TILE_DIM 4      // The dimension of the 2D tile computed by a single work-item.
#define WGS 8           // The Work-Group Size dimension (BLOCK_SIZE / TILE_DIM), which is 8.



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
    cl_command_queue compute_queue = clCreateCommandQueueWithProperties(context, device, 0, &err); CL_CHECK(err);
    cl_command_queue transfer_queue = clCreateCommandQueueWithProperties(context, device, 0, &err); CL_CHECK(err);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N, NULL, &err); CL_CHECK(err);

    float* h_Result = nullptr;
    {
        std::cout << "\n--- Running Block LU Hybrid (CPU+GPU) Implementation (Async + Pinned Memory) ---" << std::endl;

        std::string kernel_source = read_kernel_file("lu_kernels_optimized.cl");
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
        cl_kernel gemm_kernel = clCreateKernel(program, "gemm_update_kernel_register_blocked", &err); CL_CHECK(err);
        // --- Initial Data Transfer to Device (before timing) ---
        err = clEnqueueWriteBuffer(transfer_queue, d_A, CL_TRUE, 0, sizeof(float) * N * N, h_A_working, 0, NULL, NULL); CL_CHECK(err);

        // --- Create Pinned Host Buffers for Panels ---
        // CL_MEM_ALLOC_HOST_PTR tells the driver to allocate this memory in a special way
        // that the GPU's DMA engine can access directly.
        cl_mem pinned_buffers[2];
        pinned_buffers[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), NULL, &err); CL_CHECK(err);
        pinned_buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), NULL, &err); CL_CHECK(err);

        // "Map" these buffers to get host-accessible pointers. This is a cheap operation that
        // gives the CPU a standard pointer to the pinned memory region.
        float* h_panels[2];
        h_panels[0] = (float*)clEnqueueMapBuffer(transfer_queue, pinned_buffers[0], CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), 0, NULL, NULL, &err); CL_CHECK(err);
        h_panels[1] = (float*)clEnqueueMapBuffer(transfer_queue, pinned_buffers[1], CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), 0, NULL, NULL, &err); CL_CHECK(err);

        int matrix_size = N;
        clSetKernelArg(row_update_kernel, 0, sizeof(cl_mem), &d_A); clSetKernelArg(row_update_kernel, 2, sizeof(int), &matrix_size);
        clSetKernelArg(col_update_kernel, 0, sizeof(cl_mem), &d_A); clSetKernelArg(col_update_kernel, 2, sizeof(int), &matrix_size);
        clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem), &d_A); clSetKernelArg(gemm_kernel, 2, sizeof(int), &matrix_size);

        clFinish(compute_queue);
        clFinish(transfer_queue);
        auto compute_start = std::chrono::high_resolution_clock::now();

        // --- Asynchronous Pipelined Loop ---
        cl_event read_event, write_event;
        cl_event last_write_event = NULL;

        for (int k = 0; k < N; k += BLOCK_SIZE) {
            int current_panel_idx = (k / BLOCK_SIZE) % 2;

            size_t panel_device_origin[3] = { (size_t)k * sizeof(float), (size_t)k, 0 };
            size_t panel_host_origin[3]   = { 0, 0, 0 };
            size_t panel_region[3]        = { (size_t)BLOCK_SIZE * sizeof(float), (size_t)BLOCK_SIZE, 1 };

            // Step 1: Asynchronously read the current panel (k) into the pinned host buffer.
            err = clEnqueueReadBufferRect(transfer_queue, d_A, CL_FALSE, // NON-BLOCKING
                                          panel_device_origin, panel_host_origin, panel_region,
                                          (size_t)N * sizeof(float), 0,
                                          (size_t)BLOCK_SIZE * sizeof(float), 0,
                                          h_panels[current_panel_idx], 0, NULL, &read_event);
            CL_CHECK(err);

            // Step 2: If this isn't the first panel, launch GPU kernels for the PREVIOUS panel (k-1).
            // These kernels depend on the previous panel's write operation being complete.
            if (k > 0) {
                int prev_k = k - BLOCK_SIZE;
                int remaining_size = N - k; // Size of sub-matrix for this step

                clSetKernelArg(row_update_kernel, 1, sizeof(int), &prev_k);
                clSetKernelArg(col_update_kernel, 1, sizeof(int), &prev_k);
                clSetKernelArg(gemm_kernel, 1, sizeof(int), &prev_k);

                size_t global_work_size_trsm[1] = { (size_t)remaining_size };
                size_t local_work_size_trsm[1] = { BLOCK_SIZE };

                size_t gemm_global_work_size[2] = { (size_t)remaining_size / TILE_DIM, (size_t)remaining_size / TILE_DIM };
                size_t gemm_local_work_size[2]  = { WGS, WGS }; // This will now be {16, 16}

                cl_event update_events[2];

                // Kernels wait on the previous write event, and are enqueued on the compute_queue
                err = clEnqueueNDRangeKernel(compute_queue, row_update_kernel, 1, NULL, global_work_size_trsm, local_work_size_trsm, 1, &last_write_event, &update_events[0]); CL_CHECK(err);
                err = clEnqueueNDRangeKernel(compute_queue, col_update_kernel, 1, NULL, global_work_size_trsm, local_work_size_trsm, 1, &last_write_event, &update_events[1]); CL_CHECK(err);
                err = clEnqueueNDRangeKernel(compute_queue, gemm_kernel, 2, NULL, gemm_global_work_size, gemm_local_work_size, 2, update_events, NULL); CL_CHECK(err);

                clReleaseEvent(update_events[0]);
                clReleaseEvent(update_events[1]);
                clReleaseEvent(last_write_event); // Clean up the event we just used for dependency
            }

            // Step 3: Wait for the current panel's read to finish.
            // While the CPU was waiting, the GPU was busy with the previous iteration's kernels.
            err = clWaitForEvents(1, &read_event); CL_CHECK(err);
            clReleaseEvent(read_event); // Clean up the event

            // Step 4: Decompose the panel on the CPU.
            cpu_panel_decompose(h_panels[current_panel_idx], BLOCK_SIZE);

            // Step 5: Asynchronously write the decomposed panel (k) back to the GPU from pinned memory.
            err = clEnqueueWriteBufferRect(transfer_queue, d_A, CL_FALSE, // NON-BLOCKING
                                           panel_device_origin, panel_host_origin, panel_region,
                                           (size_t)N * sizeof(float), 0,
                                           (size_t)BLOCK_SIZE * sizeof(float), 0,
                                           h_panels[current_panel_idx], 0, NULL, &write_event);
            CL_CHECK(err);
            last_write_event = write_event; // Save this event for the next iteration's kernels
        }

        // No kernels need to be launched after the loop because the last GEMM update
        // corresponds to k = N - 2*BLOCK_SIZE, and its trailing submatrix has size BLOCK_SIZE.
        // The last panel decomposition is for k = N - BLOCK_SIZE, which has no trailing submatrix to update.
        if (last_write_event) clReleaseEvent(last_write_event);

        clFinish(compute_queue);
        clFinish(transfer_queue);
        auto compute_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> compute_time = compute_end - compute_start;
        double flops = (2.0 / 3.0) * pow(N, 3);
        double gflops_ocl = (flops / compute_time.count()) / 1e9;
        printf("Computation Time: %f seconds\n", compute_time.count());
        printf("Computation Performance: %.2f GFLOPS\n", gflops_ocl);

        std::cout << "Reading result from GPU for verification..." << std::endl;
        h_Result = new float[N * N];
        err = clEnqueueReadBuffer(transfer_queue, d_A, CL_TRUE, 0, sizeof(float) * N * N, h_Result, 0, NULL, NULL);
        CL_CHECK(err);

        // --- Cleanup for Pinned Buffers and Kernels ---
        clEnqueueUnmapMemObject(transfer_queue, pinned_buffers[0], h_panels[0], 0, NULL, NULL);
        clEnqueueUnmapMemObject(transfer_queue, pinned_buffers[1], h_panels[1], 0, NULL, NULL);
        clFinish(transfer_queue); // Ensure unmap is complete before release
        clReleaseMemObject(pinned_buffers[0]);
        clReleaseMemObject(pinned_buffers[1]);

        clReleaseKernel(row_update_kernel);
        clReleaseKernel(col_update_kernel);
        clReleaseKernel(gemm_kernel);
        clReleaseProgram(program);
    }

    clReleaseMemObject(d_A);
    clReleaseCommandQueue(compute_queue);
    clReleaseCommandQueue(transfer_queue);
    clReleaseContext(context);
    delete[] h_A_original;
    delete[] h_A_working;
    return 0;
}