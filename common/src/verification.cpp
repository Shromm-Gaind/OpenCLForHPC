//
// Created by Shromm
//

#include "verification.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <thread>

void verify_lu_decomposition(float* original_A, float* result_LU, int N) {
    std::cout << "\n--- Verifying LU Decomposition Result ---" << std::endl;

    float* L = new float[(size_t)N * N]();
    float* U = new float[(size_t)N * N]();
    float* P = new float[(size_t)N * N]();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i > j) {
                L[i * N + j] = result_LU[i * N + j];
            } else {
                U[i * N + j] = result_LU[i * N + j];
                if (i == j) L[i * N + j] = 1.0f;
            }
        }
    }

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "Performing L*U multiplication on CPU using " << num_threads << " threads..." << std::endl;
    std::vector<std::thread> threads;
    int rows_per_thread = N / num_threads;

    auto worker = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += L[i * N + k] * U[k * N + j];
                }
                P[i * N + j] = sum;
            }
        }
    };

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? N : start_row + rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    for (auto& t : threads) { t.join(); }

    float max_error = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        max_error = std::max(max_error, std::abs(original_A[i] - P[i]));
    }

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Verification complete. Maximum Absolute Error |A - L*U|: " << max_error << std::endl;
    if (max_error > 1e-0) {
        std::cout << "WARNING: Error is large. Result may be incorrect." << std::endl;
    } else {
        std::cout << "SUCCESS: The result is numerically correct." << std::endl;
    }

    delete[] L;
    delete[] U;
    delete[] P;
}