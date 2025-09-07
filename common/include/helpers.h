//written by Shromm Gaind
#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <CL/cl.h>

// Macro to check for OpenCL errors
#define CL_CHECK(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << err << " at line " << __LINE__ << " (" << #err << ")" << std::endl; \
        exit(1); \
    }

// Reads an entire file into a string (used for kernels)
inline std::string read_kernel_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file " << filename << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Initializes a matrix with random values and makes it diagonally dominant
inline void initialize_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < N; ++i) {
        mat[i * N + i] += (float)N;
    }
}

inline void cpu_panel_decompose(float* panel, int size) {
    for (int k = 0; k < size; k++) {
        for (int j = k; j < size; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += panel[k * size + p] * panel[p * size + j];
            panel[k * size + j] -= sum;
        }
        if (fabs(panel[k * size + k]) < 1e-9) { /* In a real app, handle this error */ }
        float pivot_inv = 1.0f / panel[k * size + k];
        for (int i = k + 1; i < size; i++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += panel[i * size + p] * panel[p * size + k];
            panel[i * size + k] = (panel[i * size + k] - sum) * pivot_inv;
        }
    }
}


#endif // HELPERS_H