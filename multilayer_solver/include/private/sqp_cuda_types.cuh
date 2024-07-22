/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sqp_cuda_types.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 09-Apr-2024 10:22:26
 */

#ifndef SQP_CUDA_TYPES_H
#define SQP_CUDA_TYPES_H

/* Include Files */

#ifdef __cplusplus
extern "C" {
#endif

/* Type Definitions */
#ifndef typedef_struct0_T
#define typedef_struct0_T
typedef struct {
  float iterations;
  float funcCount;
  char algorithm[3];
  float constrviolation;
  float stepsize;
  float lssteplength;
  float firstorderopt;
} struct0_T;
#endif /* typedef_struct0_T */

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for sqp_cuda_types.h
 *
 * [EOF]
 */
