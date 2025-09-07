//created by Shromm

__kernel void lu_row_kernel(__global float* A, const int k, const int n) {
    // Each work-item computes one element 'j' in the k-th row of U
    int j = k + get_global_id(0);

    float sum = 0.0f;
    for (int p = 0; p < k; p++) {
        sum += A[k * n + p] * A[p * n + j];
    }
    A[k * n + j] = A[k * n + j] - sum;
}

__kernel void lu_column_kernel(__global float* A, const int k, const int n) {
    // Each work-item computes one element 'i' in the k-th column of L
    int i = k + 1 + get_global_id(0);

    // This kernel can only run AFTER the k-th row kernel is complete
    // because it needs the updated value of A[k*n + k] (the pivot).
    float pivot_inv = 1.0f / A[k * n + k];

    float sum = 0.0f;
    for (int p = 0; p < k; p++) {
        sum += A[i * n + p] * A[p * n + k];
    }
    A[i * n + k] = (A[i * n + k] - sum) * pivot_inv;
}