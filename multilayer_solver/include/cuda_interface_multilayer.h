// Copyright 2024, ZHAW Zürcher Hochschule für Angewandte Wissenschaften
/*
 *           ______   _    ___        __          ___ ____   ____
 *          |__  / | | |  / \ \      / /         |_ _/ ___| / ___|
 *            / /| |_| | / _ \ \ /\ / /   _____   | |\___ \| |
 *           / /_|  _  |/ ___ \ V  V /   |_____|  | | ___) | |___
 *          /____|_| |_/_/   \_\_/\_/            |___|____/ \____|
 *
 *                Zurich University of Applied Sciences
 *         Institute of Signal Processing and Wireless Communications
 *
 * ----------------------------------------------------------------------------
 */
/**
 * @file 	cuda_interface_multilayer.h
 * @brief	Interface to the cuda library to solve the multilayer problem
 *
 * @version	1.0
 */

#ifndef MODULES_MULTILAYER_SOLVER_INCLUDE_CUDA_INTERFACE_MULTILAYER_H_
#define MODULES_MULTILAYER_SOLVER_INCLUDE_CUDA_INTERFACE_MULTILAYER_H_

#include <cuda_runtime.h>

#include <cstdlib>

#include "mini_tof_typedef.h"

namespace mlt {

class CudaInterfaceMultilayer {
 public:
  CudaInterfaceMultilayer(int n_pixels, int n_frequencies, int n_particles);
  ~CudaInterfaceMultilayer();

  /**
   * @brief Add a new image to the solver
   *
   * Converts the image to the correct format and adds it to the solver
   * @param images The image to add
   */
  void AddNewImage(const mini_tof::MiniTofImages &images);

  /**
   * @brief Solve the multilayer problem
   * @param min_amplitude Values below this amplitude are set to zero
   * @return The solved, sorted and converted image
   */
  mini_tof::MiniTofImages SolveMultilayerProblem(int min_amplitude);

 private:
  const int kNPixels_;
  const int kNFrequencies_;
  const int kNParticles_;  // PSO particles (not used in algorithm)

  int n_iterations_pso_ = 100;
  float optimality_tolerance_gradient_decent_ = 1e-3;

  float *random_numbers_ = nullptr;
  float *result_image_ = nullptr;
  float *frequencies_ = nullptr;
  float *input_image_ = nullptr;
  float *intermediate_image_arr_computing_ = nullptr;
  std::intmax_t timestamp_ = 0;
  float temperature_ = 0;

  const size_t kNRandNumbers_ =
      kNPixels_ * kNParticles_ * 16;  // n pixel * n particles * 2*(path
                                      // and amplitude) random start values

  cudaStream_t cuda_stream_ = cudaStream_t();
};

}  // namespace mlt

#endif  //MODULES_MULTILAYER_SOLVER_INCLUDE_CUDA_INTERFACE_MULTILAYER_H_
