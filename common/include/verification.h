//
// Created by Shromm
//

#ifndef VERIFICATION_H
#define VERIFICATION_H

// Verifies the LU decomposition result by calculating max|A - L*U|
void verify_lu_decomposition(float* original_A, float* result_LU, int N);

#endif // VERIFICATION_H