#include <cuda_runtime.h>

#include <cfloat>
#include <iostream>

#include "private/cuda_pso.cuh"
#include "private/sqp_cuda.cuh"

__device__ __shared__ float best_fitness_value[NUMBER_PARTICLES];
__device__ __shared__ float global_best_fitness_value;
__device__ __shared__ float
    global_best_position[4];  //{distance1,amplitude1,distance2,amplitude2}
__device__ __shared__ int best_thread_id;

__device__ const float kPiDivC = 4 * PI_F / 299792458.f;

/**
 * @brief Initializes the PSO variables for each particle
 * @param random_numbers pointer to the random numbers
 * @param c_id combined ID of the particle and pixel
 * @param current_pb_pos pointer to the personal best position
 * @param current_velo pointer to the velocity
 * @param current_pos pointer to the current position
 */
__device__ void InitPsoVariables(const float *random_numbers, unsigned int c_id,
                                 float *current_pb_pos, float *current_velo,
                                 float *current_pos);

/**
 * @brief Calculates the phase from the distance and frequency
 * @param distance distance in meter
 * @param frequency frequency in Hz
 * @return phase in radians
 */
__device__ __forceinline__ float PhaseFromDistance(float distance,
                                                   float frequency);

/**
 * @brief Limit the position to the bounds
 * @param current_pos[4] current position to be limited [distance1, amplitude1,
 * distance2, amplitude2]
 * @param current_velo[4] current velocity to be limited [distance1, amplitude1,
 * distance2, amplitude2]
 */
__device__ void LimitPsoToBounds(float *current_pos, float *current_velo);

/**
 * @brief Calculate the fitness value at the current position
 * @param frequencies pointer to the frequencies
 * @param meas_complex pointer to the measured complex values
 * @param current_pos pointer to the current position
 * @return the fitness value
 */
__device__ float CalculateFitnessValue(const float *frequencies,
                                       const bool *valid_frequency,
                                       const float meas_complex[][2],
                                       const float *current_pos);

/**
 * @brief The PSO kernel
 * @param imag_a pointer to the input image array
 * @param result_image pointer to the result image array
 * @param frequencies pointer to the frequencies
 * @param random_numbers pointer to the random numbers
 * @param n_iterations the number of iterations
 */
__global__ void Pso(const float *imag_a, float *result_image,
                    const float *frequencies, const float *random_numbers,
                    int n_iterations);

/**
 * @brief The SQP solver kernel
 * @param imag_a pointer to the input image array
 * @param result_image pointer to the result image array
 * @param starting_points pointer to the starting points
 * @param frequencies pointer to the frequencies
 * @param optimality_tolerance tollerance for the stop criterion (1=no run)
 */
__global__ void SqpSolver(const float *imag_a, float *result_image,
                          const float *starting_points,
                          const float *frequencies, float optimality_tolerance);

void PsoWrapper(const float *image_array, float *intermediate_array,
                float *result_image_wrapper, float *frequencies, float *random,
                cudaStream_t stream, int num_particles, int n_iterations_pso,
                float optimality_tolerance_gradient_decent) {
  dim3 thr_per_blk = NUMBER_PARTICLES;
  dim3 blk_in_grid = NUMBER_PIXELS;

  // Create a CUDA event
  cudaEvent_t event;
  cudaEventCreate(&event);

  Pso<<<blk_in_grid, thr_per_blk, 1, stream>>>(
      image_array, intermediate_array, frequencies, random, n_iterations_pso);

  // Record the event at the end of the Pso kernel
  cudaEventRecord(event, stream);

  // Make the GradientDecent kernel wait until the Pso kernel has finished
  cudaStreamWaitEvent(stream, event, 0);

  dim3 blks_gradient_decent = ceil((float)NUMBER_PIXELS / 32.0);

  SqpSolver<<<blks_gradient_decent, 32, 1, stream>>>(
      image_array, result_image_wrapper, intermediate_array, frequencies,
      optimality_tolerance_gradient_decent);

  // Destroy the event
  cudaEventDestroy(event);
}

__global__ void Pso(const float *imag_a, float *result_image,
                    const float *frequencies, const float *random_numbers,
                    int n_iterations) {
  //-------------thread local variables----------------------------------------

  // extract phase array with the length of the number of frequencies
  // expected to be 5. The array is sorted by the frequency and contains a
  // distance in meter and an amplitude

  unsigned int b_id = blockIdx.x;   // blockIdx in grid
  unsigned int t_id = threadIdx.x;  // threadIdx in bock
  unsigned int c_id =
      t_id + b_id * blockDim.x;  // combined ID, index of entire data array

  float meas_complex[NUMBER_FREQUENCIES][2];
  bool valid_frequency[NUMBER_FREQUENCIES];
  int n_valid_freq = 0;
  float exception_code = 0;
  float amplitude = 0;

  if (b_id < NUMBER_PIXELS) {
    //-------------------------block wide variables-----------------------------
    float amp_info_block[NUMBER_FREQUENCIES];
    float phase_inf_block[NUMBER_FREQUENCIES];

    // load phase and amplitude information from global  memory in thread memory
    phase_inf_block[0] = imag_a[b_id];
    amp_info_block[0] = imag_a[b_id + (NUMBER_PIXELS)];
    phase_inf_block[1] = imag_a[b_id + (NUMBER_PIXELS * 2)];
    amp_info_block[1] = imag_a[b_id + (NUMBER_PIXELS * 3)];
    phase_inf_block[2] = imag_a[b_id + (NUMBER_PIXELS * 4)];
    amp_info_block[2] = imag_a[b_id + (NUMBER_PIXELS * 5)];
    phase_inf_block[3] = imag_a[b_id + (NUMBER_PIXELS * 6)];
    amp_info_block[3] = imag_a[b_id + (NUMBER_PIXELS * 7)];
    phase_inf_block[4] = imag_a[b_id + (NUMBER_PIXELS * 8)];
    amp_info_block[4] = imag_a[b_id + (NUMBER_PIXELS * 9)];

    // calculate the complex measurement values (from amp/phase to complex
    // real/imag)
    float phase_cos;
    float phase_sin;

    for (int m = 0; m < NUMBER_FREQUENCIES; m++) {
      __sincosf(phase_inf_block[m], &phase_sin, &phase_cos);
      meas_complex[m][0] = amp_info_block[m] * phase_cos;
      meas_complex[m][1] = amp_info_block[m] * phase_sin;
      valid_frequency[m] =
          (phase_inf_block[m] <
           (MIN_INVALID_VALUE - 1));  // -1 to avoid rounding errors
      if (!valid_frequency[m]) {
        exception_code = phase_inf_block[m];
        amplitude = amp_info_block[m];
      }
      n_valid_freq += valid_frequency[m];
    }
  }

  if (n_valid_freq >= 3) {
    if (c_id < NUMBER_PIXELS * NUMBER_PARTICLES) {
      //----------------init thread local variables----------------
      float current_pb_pos[4];
      float current_velo[4];
      float current_pos[4];

      float personal_best_eval = FLT_MAX;
      float current_eval = 0;
      int stall_counter = 0;
      float inertia_weight = INERTIA_WEIGHT;

      int local_best_thread_id;

      InitPsoVariables(random_numbers, c_id, current_pb_pos, current_velo,
                       current_pos);

      if (t_id == 0) {
        global_best_fitness_value = CalculateFitnessValue(
            frequencies, valid_frequency, meas_complex, current_pos);
      }

      // here the iterations for the PSO start
      for (int i = 0; i < n_iterations; ++i) {
        current_eval = CalculateFitnessValue(frequencies, valid_frequency,
                                             meas_complex, current_pos);

        // each particle writes its new fitness value into the shared block
        // memory if it is better than the old one
        if (current_eval < personal_best_eval) {
          best_fitness_value[t_id] = current_eval;
          personal_best_eval = current_eval;
          stall_counter = max(0, stall_counter - 1);
          if (stall_counter < 2) {
            inertia_weight *= 2;
            inertia_weight = fminf(inertia_weight, 1.1);
          } else if (stall_counter > 5) {
            inertia_weight /= 2;
            inertia_weight = fmaxf(inertia_weight, 0.1);
          }
          current_pb_pos[0] = current_pos[0];
          current_pb_pos[1] = current_pos[1];
          current_pb_pos[2] = current_pos[2];
          current_pb_pos[3] = current_pos[3];
        } else {
          ++stall_counter;
        }
        __syncthreads();  // wait until all threads for same pixel have finished
                          // above code and updated the personal best fitness
                          // value in shared mem

        // thread 3 gets the best fitness value out of all particles personal
        // best values and shares it to the other particles
        // (global_best_fitness_value) also the thread which had the best
        // fitness value is identified and shared (best_thread_id)
        //  gls is used so that the comparing thread doesn't need to compare the
        //        fitness values to a shared variable (it's faster)
        if (t_id == 3) {
          local_best_thread_id = -1;

          auto gls = global_best_fitness_value;

          for (int16_t k = 0; k < NUMBER_PARTICLES; ++k) {
            if (gls > best_fitness_value[k]) {
              gls = best_fitness_value[k];
              local_best_thread_id = k;
            }
          }
          best_thread_id = local_best_thread_id;
          global_best_fitness_value = gls;
        }

        __syncthreads();

        // Update global best position: take best performing particle position
        // and write it to block shared memory
        if (t_id == best_thread_id) {
          global_best_position[0] = current_pos[0];
          global_best_position[1] = current_pos[1];
          global_best_position[2] = current_pos[2];
          global_best_position[3] = current_pos[3];
        }
        __syncthreads();  // wait until all threads have finished above code and
                          // updated the global best position in shared memory

        // update velocity
        // NOTE: the random numbers are used to make the velocity update random,
        // we need to use predefined "random" values, since there is no
        // randomness in CUDA this is also the reason why we don't do this in a
        // for loop, because we want different random numbers for each velocity
        // update

        current_velo[0] = current_velo[0] * inertia_weight +
                          COGNITIVE_WEIGHT * random_numbers[i + c_id] *
                              (current_pb_pos[0] - current_pos[0]) +
                          SOCIAL_WEIGHT * random_numbers[i + 1 + c_id] *
                              (global_best_position[0] - current_pos[0]);
        current_velo[1] = current_velo[1] * inertia_weight +
                          COGNITIVE_WEIGHT * random_numbers[i + 2 + c_id] *
                              (current_pb_pos[1] - current_pos[1]) +
                          SOCIAL_WEIGHT * random_numbers[i + 3 + c_id] *
                              (global_best_position[1] - current_pos[1]);
        current_velo[2] = current_velo[2] * inertia_weight +
                          COGNITIVE_WEIGHT * random_numbers[i + 4 + c_id] *
                              (current_pb_pos[2] - current_pos[2]) +
                          SOCIAL_WEIGHT * random_numbers[i + 5 + c_id] *
                              (global_best_position[2] - current_pos[2]);
        current_velo[3] = current_velo[3] * inertia_weight +
                          COGNITIVE_WEIGHT * random_numbers[i + 6 + c_id] *
                              (current_pb_pos[3] - current_pos[3]) +
                          SOCIAL_WEIGHT * random_numbers[i + 7 + c_id] *
                              (global_best_position[3] - current_pos[3]);

        // update position
        current_pos[0] += current_velo[0];
        current_pos[1] += current_velo[1];
        current_pos[2] += current_velo[2];
        current_pos[3] += current_velo[3];

        LimitPsoToBounds(current_pos, current_velo);
      }

      __syncthreads();

      if (t_id == 0) {
        result_image[b_id] = global_best_position[0];
        result_image[b_id + NUMBER_PIXELS] = global_best_position[1];
        result_image[b_id + 2 * NUMBER_PIXELS] = global_best_position[2];
        result_image[b_id + 3 * NUMBER_PIXELS] = global_best_position[3];
      }
    }
  } else {
    if (t_id == 0) {
      result_image[b_id] = exception_code;
      result_image[b_id + NUMBER_PIXELS] = amplitude;
      result_image[b_id + 2 * NUMBER_PIXELS] = exception_code;
      result_image[b_id + 3 * NUMBER_PIXELS] = amplitude;
    }
  }
}

__global__ void SqpSolver(const float *imag_a, float *result_image,
                          const float *starting_points,
                          const float *frequencies,
                          float optimality_tolerance) {
  unsigned int c_id =
      threadIdx.x +
      blockIdx.x * blockDim.x;  // combined ID, index of entire data array

  if (optimality_tolerance < 1) {  // skipp sqp solver by setting optimtol > 1
    if (c_id < NUMBER_PIXELS) {
      float sol[4];
      float x0[4];
      const float kLb[4] = {MIN_AMPLITUDE, MIN_AMPLITUDE, MIN_DISTANCE,
                            MIN_DISTANCE};
      const float kUp[4] = {MAX_AMPLITUDE, MAX_AMPLITUDE, MAX_DISTANCE,
                            MAX_DISTANCE};
      float meas_complex[NUMBER_FREQUENCIES * 2];
      float eflag;
      float fval;
      struct0_T output;

      float amp_info_block[NUMBER_FREQUENCIES];
      float phase_inf_block[NUMBER_FREQUENCIES];
      bool valid_frequency[NUMBER_FREQUENCIES];
      int n_valid_freq = 0;
      float exception_code = 0;
      float amplitude = 0;

      phase_inf_block[0] = imag_a[c_id];
      amp_info_block[0] = imag_a[c_id + (NUMBER_PIXELS)];
      phase_inf_block[1] = imag_a[c_id + (NUMBER_PIXELS * 2)];
      amp_info_block[1] = imag_a[c_id + (NUMBER_PIXELS * 3)];
      phase_inf_block[2] = imag_a[c_id + (NUMBER_PIXELS * 4)];
      amp_info_block[2] = imag_a[c_id + (NUMBER_PIXELS * 5)];
      phase_inf_block[3] = imag_a[c_id + (NUMBER_PIXELS * 6)];
      amp_info_block[3] = imag_a[c_id + (NUMBER_PIXELS * 7)];
      phase_inf_block[4] = imag_a[c_id + (NUMBER_PIXELS * 8)];
      amp_info_block[4] = imag_a[c_id + (NUMBER_PIXELS * 9)];

      float phase_cos;
      float phase_sin;
      for (int m = 0; m < NUMBER_FREQUENCIES; m++) {
        __sincosf(phase_inf_block[m], &phase_sin, &phase_cos);
        meas_complex[m] = amp_info_block[m] * phase_cos;
        meas_complex[m + NUMBER_FREQUENCIES] = amp_info_block[m] * phase_sin;
        valid_frequency[m] =
            (phase_inf_block[m] <
             (MIN_INVALID_VALUE - 1));  // -1 to avoid rounding errors
        if (!valid_frequency[m]) {
          exception_code = phase_inf_block[m];
          amplitude = amp_info_block[m];
        }
        n_valid_freq += valid_frequency[m];
      }
      if (n_valid_freq >= 3) {
        x0[0] = starting_points[c_id + 1 * NUMBER_PIXELS];
        x0[1] = starting_points[c_id + 3 * NUMBER_PIXELS];
        x0[2] = starting_points[c_id + 0 * NUMBER_PIXELS];
        x0[3] = starting_points[c_id + 2 * NUMBER_PIXELS];

        SqpCuda(x0, kLb, kUp, meas_complex, frequencies, valid_frequency, sol,
                &fval, &eflag, &output, optimality_tolerance);

        result_image[c_id] = (float)sol[2];
        result_image[c_id + NUMBER_PIXELS] = (float)sol[0];
        result_image[c_id + 2 * NUMBER_PIXELS] = (float)sol[3];
        result_image[c_id + 3 * NUMBER_PIXELS] = (float)sol[1];
      } else {
        result_image[c_id] = exception_code;
        result_image[c_id + NUMBER_PIXELS] = amplitude;
        result_image[c_id + 2 * NUMBER_PIXELS] = exception_code;
        result_image[c_id + 3 * NUMBER_PIXELS] = amplitude;
      }
    }
  } else {
    result_image[c_id] = starting_points[c_id];
    result_image[c_id + NUMBER_PIXELS] = starting_points[c_id + NUMBER_PIXELS];
    result_image[c_id + 2 * NUMBER_PIXELS] =
        starting_points[c_id + 2 * NUMBER_PIXELS];
    result_image[c_id + 3 * NUMBER_PIXELS] =
        starting_points[c_id + 3 * NUMBER_PIXELS];
  }
}
__device__ void InitPsoVariables(const float *random_numbers, unsigned int c_id,
                                 float *current_pb_pos, float *current_velo,
                                 float *current_pos) {
  // init position
  current_pos[0] = random_numbers[c_id * 4] * MAX_DISTANCE;      // distance 1
  current_pos[1] = random_numbers[c_id * 4 + 1];                 // amplitude 1
  current_pos[2] = random_numbers[c_id * 4 + 2] * MAX_DISTANCE;  // distance 2
  current_pos[3] = random_numbers[c_id * 4 + 3];                 // amplitude 2

  // init personal best position
  current_pb_pos[0] = current_pos[0];  // distance 1
  current_pb_pos[1] = current_pos[1];  // amplitude 1
  current_pb_pos[2] = current_pos[2];  // distance 2
  current_pb_pos[3] = current_pos[3];  // amplitude 2

  // init velocity
  current_velo[0] = random_numbers[c_id * 4 + 4];  // distance velocity 1
  current_velo[1] = random_numbers[c_id * 4 + 5];  // amplitude velocity 1
  current_velo[2] = random_numbers[c_id * 4 + 6];  // distance velocity 2
  current_velo[3] = random_numbers[c_id * 4 + 7];  // amplitude velocity 2
}

__device__ __forceinline__ float PhaseFromDistance(float distance,
                                                   float frequency) {
  float phase = kPiDivC * distance * frequency;
  return phase;
}

__device__ void LimitPsoToBounds(float *current_pos, float *current_velo) {
  // Define the bounds
  float bounds[4] = {MAX_DISTANCE, MAX_AMPLITUDE, MAX_DISTANCE, MAX_AMPLITUDE};

  // Loop over each dimension
  for (int i = 0; i < 4; i++) {
    if (current_pos[i] > bounds[i]) {
      current_pos[i] = bounds[i];
      current_velo[i] = 0;
    } else if (current_pos[i] < 0) {
      current_pos[i] = 0;
      current_velo[i] = 0;
    }
  }
}

__device__ float CalculateFitnessValue(const float *frequencies,
                                       const bool *valid_frequency,
                                       const float meas_complex[][2],
                                       const float *current_pos) {
  float pos_complex_1[2];         // estimated complex value for first object
                                  // (distance and amplitude)
  float pos_complex_2[2];         // estimated complex value for second object
                                  // (distance and amplitude)
  float pos_complex_combined[2];  // this is the hypothetical measurement
                                  // value (first and second obj combined)

  float current_eval = 0;  // reset the fitness value for the current particle
  // Calculate the hypothetical measurement values for the current position
  // of a particle at an all given frequencies
  for (int m = 0; m < NUMBER_FREQUENCIES; m++) {
    if (valid_frequency[m]) {
      float phase = PhaseFromDistance(current_pos[0], frequencies[m]);
      float cos_phase;
      float sin_phase;
      __sincosf(phase, &sin_phase, &cos_phase);
      pos_complex_1[0] =
          current_pos[1] * cos_phase;  // calc real part of complex
                                       // value for first object
      pos_complex_1[1] =
          current_pos[1] * sin_phase;  // calc imag part of complex
                                       // value for first object

      phase = PhaseFromDistance(current_pos[2], frequencies[m]);
      __sincosf(phase, &sin_phase, &cos_phase);
      pos_complex_2[0] =
          current_pos[3] * cos_phase;  // calc real part of complex
                                       // value for second object
      pos_complex_2[1] =
          current_pos[3] * sin_phase;  // calc imag part of complex
                                       // value for second object

      pos_complex_combined[0] =
          pos_complex_1[0] + pos_complex_2[0];  // calc real part of complex
                                                // value for combined objects
      pos_complex_combined[1] =
          pos_complex_1[1] + pos_complex_2[1];  // calc imag part of complex
                                                // value for combined objects

      // subtract the hypothetical measurement from the real measurement and
      // square it to evaluate the fitness of a particle.
      // Note: Taking the square root is not necessary
      auto eval0 = meas_complex[m][0] - pos_complex_combined[0];
      auto eval1 = meas_complex[m][1] - pos_complex_combined[1];
      current_eval += eval0 * eval0 + eval1 * eval1;
    }
  }
  return current_eval;
}
