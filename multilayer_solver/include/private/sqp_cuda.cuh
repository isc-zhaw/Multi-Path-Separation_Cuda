/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sqp_cuda.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 09-Apr-2024 10:22:26
 */

#ifndef SQP_CUDA_H
#define SQP_CUDA_H

/* Include Files */
#include "private/sqp_cuda_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
__device__ void SqpCuda(const float x0[4], const float lb[4], const float ub[4],
                        const float mmvek[10], const float b_fmod[5],
                        const bool valid_freq[5], float sol[4], float *fval,
                        float *eflag, struct0_T *output,
                        float optimality_tolerance);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for sqp_cuda.h
 *
 * [EOF]
 */
