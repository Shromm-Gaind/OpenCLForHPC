// main.cpp - Complete harness modified to test YOUR kernel as a pure GEMM

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath> // For ceil

// These headers are from the original open-source benchmark
#include "settings.hpp"
#include "utils/init_vector.hpp"
#include "utils/opencl_runtime.hpp"
#include "utils/timer.hpp"

#ifdef HAVE_CLBLAST
#include "clblast.h"
#endif

// CPU BLAS libraries for fast and accurate reference calculation
#ifdef HAVE_OPENBLAS
#include "cblas.h"
#elif HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

// The reference GEMM implementation for verification.
// Now accepts a flag to handle transposed B matrices.
void RefGemm(const int M, const int N, const int K,
             const float *A, const float *B, float *C, bool b_is_transposed) {
#if defined(HAVE_ACCELERATE) || defined(HAVE_OPENBLAS)
    auto b_transpose_flag = b_is_transposed ? CblasTrans : CblasNoTrans;
    int ldb = b_is_transposed ? K : N;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, b_transpose_flag, M, N, K, 1.f, A, K, B, ldb, 0.f, C, N);
#else
    if (b_is_transposed) {
        std::cerr << "Warning: Simple triple-loop reference does not support transposed B. Verification might be incorrect." << std::endl;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float c_val = 0;
            for (int k = 0; k < K; k++) {
                c_val += A[i * K + k] * B[k * N + j]; // This assumes non-transposed B
            }
            C[i * N + j] = c_val;
        }
    }
#endif
}

// Function to compare the GPU result with the reference CPU result
float Compare(const int M, const int N, const float *C, const float *C_ref) {
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
    }
    return max_diff;
}

void TransposeMatrix(const int rows, const int cols, const std::vector<float>& input, std::vector<float>& output) {
    output.resize(cols * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}


int worker(OpenCLRuntime& runtime, const int M, const int N, const int K,
           const int test_num_repeats, const std::string &kernel_file) {

    cl::Kernel kernel;
    cl::NDRange global_sizes, local_sizes;

    // BUG FIX 2: Declare B_device and C_device here to ensure they are in scope
    // for all kernel configurations and for the final read-back call.
    cl::Buffer B_device, C_device;

    // --- Data Preparation (Common part) ---
    std::cout << "Preparing host data..." << std::endl;
    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N), C_ref(M * N);
    InitVector(A_host);
    InitVector(B_host);

    auto A_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, A_host.size() * sizeof(float));
    runtime.GetCommandQueue().enqueueWriteBuffer(A_device, CL_TRUE, 0, A_host.size() * sizeof(float), A_host.data());
    C_device = cl::Buffer(runtime.GetContext(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, C_host.size() * sizeof(float));

    // --- KERNEL-SPECIFIC SETUP ---
    if (kernel_file.find("my_gemm_improved.cl") != std::string::npos) {
        std::cout << "Configuring for my_gemm_improved.cl..." << std::endl;

        // Transpose B and create its device buffer
        std::vector<float> B_host_transposed;
        TransposeMatrix(K, N, B_host, B_host_transposed);
        B_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, B_host_transposed.size() * sizeof(float));
        runtime.GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host_transposed.size() * sizeof(float), B_host_transposed.data());

        // Define tuning parameters
        const int MWG = 32, NWG = 32, KWG = 32;
        const int MWI = 4, NWI = 4;
        const int WGS_X = NWG / NWI; // 8
        const int WGS_Y = MWG / MWI; // 8
        std::string build_options =
                "-D MWG=" + std::to_string(MWG) + " -D NWG=" + std::to_string(NWG) + " -D KWG=" + std::to_string(KWG) +
                " -D MWI=" + std::to_string(MWI) + " -D NWI=" + std::to_string(NWI) +
                " -D WGS_X=" + std::to_string(WGS_X) + " -D WGS_Y=" + std::to_string(WGS_Y);
        kernel = runtime.BuildKernel(kernel_file, build_options, "my_gemm_improved");

        kernel.setArg(0, M); kernel.setArg(1, N); kernel.setArg(2, K);
        kernel.setArg(3, A_device); kernel.setArg(4, B_device); kernel.setArg(5, C_device);

        local_sizes = cl::NDRange(WGS_X, WGS_Y);

        // BUG FIX 1: This is the correct global size calculation.
        // We need enough WORK-GROUPS to cover the matrix, not enough threads.
        size_t global_x = (size_t)ceil((float)N / NWG) * WGS_X;
        size_t global_y = (size_t)ceil((float)M / MWG) * WGS_Y;
        global_sizes = cl::NDRange(global_x, global_y);

    }
    else { // --- SETUP FOR OTHER KERNELS ---
        B_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, B_host.size() * sizeof(float));
        runtime.GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());

        if (kernel_file.find("opencv_gemm.cl") != std::string::npos) {
            std::cout << "Configuring for opencv_gemm.cl..." << std::endl;

            B_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, B_host.size() * sizeof(float));
            runtime.GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());

            size_t maxWorkGroupSize = runtime.GetMaxWorkGroupSize();
            int cn = 1, kercn = 4;
            int block_size = (maxWorkGroupSize >= 1024) ? 32 : 16;

            std::string build_options = "-DT=float -DT1=float -DWT=float4 -Dcn=" + std::to_string(cn) +
                                        " -Dkercn=" + std::to_string(kercn) +
                                        " -DLOCAL_SIZE=" + std::to_string(block_size);
            if ((N * cn / kercn) % block_size != 0) {
                build_options += " -D NO_MULT";
            }
            kernel = runtime.BuildKernel(kernel_file, build_options, "gemm");

            kernel.setArg(0, A_device);
            kernel.setArg(1, K * (int)sizeof(float)); // A_step
            kernel.setArg(2, 0);                      // A_offset
            kernel.setArg(3, B_device);
            kernel.setArg(4, N * (int)sizeof(float)); // B_step
            kernel.setArg(5, 0);                      // B_offset
            kernel.setArg(6, C_device);
            kernel.setArg(7, N * (int)sizeof(float)); // D_step
            kernel.setArg(8, 0);                      // D_offset
            kernel.setArg(9, M);                      // D_rows
            kernel.setArg(10, (N * cn) / kercn);      // D_cols
            kernel.setArg(11, K);                     // n
            kernel.setArg(12, 1.0f);                  // alpha
            kernel.setArg(13, 0.0f);                  // beta

            global_sizes = cl::NDRange((N * cn) / kercn, M);
            local_sizes = cl::NDRange(block_size, block_size);

        }
        else if (kernel_file.find("my_gemm_kernel.cl") != std::string::npos) {
            std::cout << "Configuring for my_gemm_kernel.cl..." << std::endl;
            kernel = runtime.BuildKernel(kernel_file, "", "my_gemm_pure_register_blocked");

            kernel.setArg(0, M); kernel.setArg(1, N); kernel.setArg(2, K);
            kernel.setArg(3, A_device); kernel.setArg(4, B_device); kernel.setArg(5, C_device);

            const int TILE_DIM = 4; // This is the tile per thread, not group
            const int WGS = 8;

            // This kernel's grid is based on tiles-per-thread
            const int BLOCK_SIZE = WGS * TILE_DIM;
            global_sizes = cl::NDRange((size_t)N/TILE_DIM, (size_t)M/TILE_DIM);
            local_sizes = cl::NDRange(WGS, WGS);
        } else {
            std::cerr << "Unsupported kernel file in main.cpp!" << std::endl;
            return 0;
        }
    }
    runtime.GetCommandQueue().finish();

    // --- Run and Benchmark Kernel ---
    std::cout << "Starting benchmark runs..." << std::endl;
    cl::Event event;
    TickMeter meter;
    meter.Reset();
    runtime.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
    event.wait();

    for (int i = 0; i < test_num_repeats; i++) {
        meter.Start();
        runtime.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
        event.wait();
        meter.End();
    }

    std::cout << "Reading results from device..." << std::endl;
    runtime.GetCommandQueue().enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    // --- Verification and Results ---
    std::cout << "Verifying results against CPU reference..." << std::endl;
    // For verification, ALWAYS use the original non-transposed B matrix.
    RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data(), false);
    float max_diff = Compare(M, N, C_host.data(), C_ref.data());

    double mean, median, minimum;
    meter.GetPerformanceResults(mean, median, minimum);
    double gflops = (2.0 * (double)M * (double)N * (double)K * 1.0e-6) / mean;

    printf("\n--- RESULTS ---\n");
    printf("Kernel: %s\nM=%d, N=%d, K=%d\nMean Time: %.4fms, Min Time: %.4fms\nGFLOPS: %.2f\nMax Diff: %f\n",
           kernel_file.c_str(), M, N, K, mean, minimum, gflops, max_diff);
    printf("---------------\n");

    return 1;
}

#ifdef HAVE_CLBLAST
// CLBlast worker remains unchanged
int worker_clblast(OpenCLRuntime& runtime, const int M, const int N, const int K,
                   const int test_num_repeats) {
    std::cout << "Preparing host data for CLBlast..." << std::endl;
    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N), C_ref(M * N);
    InitVector(A_host); InitVector(B_host);

    std::cout << "Transferring data to device..." << std::endl;
    auto A_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, A_host.size() * sizeof(float));
    auto B_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, B_host.size() * sizeof(float));
    auto C_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, C_host.size() * sizeof(float));
    runtime.GetCommandQueue().enqueueWriteBuffer(A_device, CL_TRUE, 0, A_host.size() * sizeof(float), A_host.data());
    runtime.GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());
    runtime.GetCommandQueue().finish();

    std::cout << "Starting CLBlast benchmark runs..." << std::endl;
    auto queue_raw = runtime.GetCommandQueue()();
    TickMeter meter;

    cl_event event_raw;
    clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                  M, N, K, 1.0f, A_device(), 0, K, B_device(), 0, N, 0.0f, C_device(), 0, N, &queue_raw, &event_raw);
    clWaitForEvents(1, &event_raw);
    clReleaseEvent(event_raw);

    meter.Reset();
    for (int i = 0; i < test_num_repeats; i++) {
        meter.Start();
        cl_event timed_event_raw;
        clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                      M, N, K, 1.0f, A_device(), 0, K, B_device(), 0, N, 0.0f, C_device(), 0, N, &queue_raw, &timed_event_raw);
        clWaitForEvents(1, &timed_event_raw);
        clReleaseEvent(timed_event_raw);
        meter.End();
    }

    std::cout << "Reading results from device..." << std::endl;
    runtime.GetCommandQueue().enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    double mean, median, minimum;
    meter.GetPerformanceResults(mean, median, minimum);
    double gflops = (2.0 * (double)M * (double)N * (double)K * 1.0e-6) / mean;
    RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data(), false); // B is not transposed here
    float max_diff = Compare(M, N, C_host.data(), C_ref.data());

    printf("\n--- RESULTS ---\n");
    printf("Kernel: CLBlast Library\nM=%d, N=%d, K=%d\nMean Time: %.4fms, Min Time: %.4fms\nGFLOPS: %.2f\nMax Diff: %f\n",
           M, N, K, mean, minimum, gflops, max_diff);
    printf("---------------\n");

    return 1;
}
#endif

int main(int argc, char **argv) {
    OpenCLRuntime runtime;
    std::cout << "Platform: " << runtime.GetPlatformName() << std::endl
              << "Device:   " << runtime.GetDeviceName() << std::endl;

    const int matrix_size = 8192; // Adjusted for quicker testing
    const int M = matrix_size, N = matrix_size, K = matrix_size;
    const int test_num_repeats = 10;

    // =========================================================================
    // ===                 CHOOSE WHICH TEST TO RUN HERE                     ===
    // =========================================================================
    // 1: Your original kernel (my_gemm_kernel.cl)
    // 2: The OpenCV shared-memory-focused kernel
    // 3: The CLBlast library

    int test_selection = 1;

    // =========================================================================

    switch(test_selection) {
        case 1: {
            std::string kernel_file = "kernels/my_gemm_kernel.cl";
            printf("\nTarget: Your Original Kernel (%s)\n", kernel_file.c_str());
            worker(runtime, M, N, K, test_num_repeats, kernel_file);
            break;
        }
        case 2: {
            std::string kernel_file = "kernels/opencv_gemm.cl";
            printf("\nTarget: OpenCV Kernel (%s)\n", kernel_file.c_str());
            worker(runtime, M, N, K, test_num_repeats, kernel_file);
            break;
        }
        case 3: {
#ifdef HAVE_CLBLAST
            printf("\nTarget: CLBlast Library\n");
            worker_clblast(runtime, M, N, K, test_num_repeats);
#else
            printf("CLBlast test selected, but library was not found during compilation.\n");
#endif
            break;
        }
        default:
            printf("Invalid test selection.\n");
            break;
    }

    return 0;
}