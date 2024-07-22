
#ifndef MODULES_GPUPSO_INCLUDE_PRIVATE_HELPER_SQP_CUH_
#define MODULES_GPUPSO_INCLUDE_PRIVATE_HELPER_SQP_CUH_

#include "private/sqp_cuda_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif
__device__ int CompareArrays(const char *a, const char *b, int size);
//__device__ boolean_T rtIsInf(real_T value);
//__device__ boolean_T rtIsInfF(real32_T value);
//__device__ boolean_T rtIsNaN(real_T value);
//__device__ boolean_T rtIsNaNF(real32_T value);

#ifdef __cplusplus
}
#endif

#endif // MODULES_GPUPSO_INCLUDE_PRIVATE_HELPER_SQP_CUH_