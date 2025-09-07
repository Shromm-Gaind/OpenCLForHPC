#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Define a reasonable matrix size for performance testing
#define N 2048

// Note: For simplicity, this code uses statically allocated matrices.
// For larger N, dynamic allocation with malloc is required.
// float (*A)[N] = malloc(sizeof(float[N][N]));
static float A[N][N];
static float b[N];
static float x[N];

// Function to initialize a matrix with random values for testing
void initialize_system() {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        b[i] = (float)rand() / (float)RAND_MAX;
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}


// Function to perform in-place Doolittle LU Decomposition
// Returns 0 on success, -1 on failure (singular matrix)
int lu_decompose(float A[N][N]) {
    for (int k = 0; k < N; k++) {
        for (int j = k; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[k][p] * A[p][j];
            }
            A[k][j] = A[k][j] - sum;
        }

        if (fabs(A[k][k]) < 1e-9) { // Use a small epsilon for floating point
            fprintf(stderr, "Matrix is singular!\n");
            return -1;
        }

        float pivot_inv = 1.0f / A[k][k];
        for (int i = k + 1; i < N; i++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i][p] * A[p][k];
            }
            A[i][k] = (A[i][k] - sum) * pivot_inv;
        }
    }
    return 0;
}

int main() {
    printf("Initializing a %d x %d system...\n", N, N);
    initialize_system();

    printf("Performing sequential LU decomposition on the CPU...\n");

    // --- TIMING START ---
    clock_t start = clock();

    // Perform the decomposition
    int success = lu_decompose(A);

    // --- TIMING END ---
    clock_t end = clock();

    if (success != 0) {
        printf("Decomposition failed.\n");
        return -1;
    }

    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU execution time: %f seconds\n", cpu_time_used);

    // --- MFLOPS CALCULATION ---
    // The number of floating point operations for LU decomposition
    // is approximately (2/3) * N^3.
    // We count one multiplication and one addition/subtraction as 2 FLOPs.
    // The loops execute roughly N^3 / 3 times, each with a mul and a sub.
    double flops = (2.0 / 3.0) * pow(N, 3);
    double mflops = (flops / cpu_time_used) / 1e6; // Convert to millions

    printf("Performance: %.2f MFLOPS\n", mflops);

    // (Optional: You could also time and include the solve_system part,
    // but its O(N^2) complexity means its contribution to runtime is
    // negligible for large N compared to the O(N^3) decomposition.)

    return 0;
}