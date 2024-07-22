
#ifndef MODULES_MULTILAYER_SOLVER_INCLUDE_PRIVATE_CUDA_PSO_CUH_
#define MODULES_MULTILAYER_SOLVER_INCLUDE_PRIVATE_CUDA_PSO_CUH_

#include <cuda_runtime.h>

#define NUMBER_PARTICLES 128
#define NUMBER_FREQUENCIES 5
#define INERTIA_WEIGHT (1.1f)
#define COGNITIVE_WEIGHT (1.4f)
#define SOCIAL_WEIGHT (1.4f)
#define NUMBER_PIXELS (60 * 160)
#define MIN_INVALID_VALUE 16000
#define MIN_DISTANCE 0.3
#define MAX_DISTANCE 4
#define MIN_AMPLITUDE 0
#define MAX_AMPLITUDE 2

#define PI_F 3.14159265358979f

/**
 * @brief The PSO wrapper that includes the PSO and the gradient decent kernel
 * @param image_array pointer to the input image array
 * @param intermediate_array pointer to the intermediate array to pas data
 * between the pso and gradient decent kernels
 * @param result_image_wrapper pointer to the result image array
 * @param frequencies pointer to the frequencies
 * @param random pointer to the random numbers
 * @param stream the CUDA stream
 * @param num_particles the number of particles
 * @param n_iterations_pso the number of iterations for the PSO
 * @param optimality_tolerance_gradient_decent tollerance for the stop criterion
 * (1=no run)
 */
void PsoWrapper(const float *image_array, float *intermediate_array,
                float *result_image_wrapper, float *frequencies, float *random,
                cudaStream_t stream, int num_particles, int n_iterations_pso,
                float optimality_tolerance_gradient_decent);

#endif  // MODULES_MULTILAYER_SOLVER_INCLUDE_PRIVATE_CUDA_PSO_CUH_