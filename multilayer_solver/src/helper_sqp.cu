#include "private/helper_sqp.cuh"

__device__ int CompareArrays(const char *a, const char *b, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i] != b[i]) {
      return 1;
    }
  }
  return 0;
}
//
///*
// * Function: rtIsInf ==================================================
// *  Abstract:
// *  Test if value is infinite
// */
//__device__ boolean_T rtIsInf(real_T value) { return (isinf(value) != 0U); }
//
///*
// * Function: rtIsInfF =================================================
// *  Abstract:
// *  Test if single-precision value is infinite
// */
//__device__ boolean_T rtIsInfF(real32_T value) {
//  return (isinf((real_T)value) != 0U);
//}
//
///*
// * Function: rtIsNaN ==================================================
// *  Abstract:
// *  Test if value is not a number
// */
//__device__ boolean_T rtIsNaN(real_T value) { return (isnan(value) != 0U); }
//
///*
// * Function: rtIsNaNF =================================================
// *  Abstract:
// *  Test if single-precision value is not a number
// */
//__device__ boolean_T rtIsNaNF(real32_T value) {
//  return (isnan((real_T)value) != 0U);
//}
