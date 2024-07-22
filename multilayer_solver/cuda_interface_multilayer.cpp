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
 * @file 	cuda_interface_multilayer.cpp
 * @brief	Interface to the cuda library to solve the multilayer problem
 *
 * @version	1.0
 */

#include "cuda_interface_multilayer.h"

#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include "mini_tof_typedef.h"
#include "private/camera_parameters.h"
#include "private/cuda_pso.cuh"

namespace mlt {

const float kPi = 3.14159265358979f;
static const int kMinDistDifference = 100;

CudaInterfaceMultilayer::CudaInterfaceMultilayer(int n_pixels,
                                                 int n_frequencies,
                                                 int n_particles)
    : kNPixels_(n_pixels),
      kNFrequencies_(n_frequencies),
      kNParticles_(NUMBER_PARTICLES) {
  auto error = cudaMallocManaged(&input_image_,
                                 kNPixels_ * 2 * kNFrequencies_ * sizeof(float),
                                 cudaMemAttachHost);
  if (error != cudaSuccess) {
    throw std::runtime_error("Error allocating (Input image): " +
                             std::string(cudaGetErrorString(error)));
  }

  error = cudaMallocManaged(&result_image_, 2 * kNPixels_ * 2 * sizeof(float));
  if (error != cudaSuccess) {
    throw std::runtime_error("Error allocating (Result image): " +
                             std::string(cudaGetErrorString(error)));
  }

  error = cudaMalloc(&intermediate_image_arr_computing_,
                     kNPixels_ * 4 * sizeof(float));
  if (error != cudaSuccess) {
    throw std::runtime_error("Error allocating (Result image): " +
                             std::string(cudaGetErrorString(error)));
  }

  error = cudaMallocManaged(&frequencies_, sizeof(float) * kNFrequencies_,
                            cudaMemAttachHost);
  if (error != cudaSuccess) {
    throw std::runtime_error("Error allocating (Frequency): " +
                             std::string(cudaGetErrorString(error)));
  }

  error = cudaMalloc(&random_numbers_, kNRandNumbers_ * sizeof(float));
  if (error != cudaSuccess) {
    throw std::runtime_error("Error allocating (rand numbers): " +
                             std::string(cudaGetErrorString(error)));
  }

  cudaStreamCreate(&cuda_stream_);

  curandGenerator_t rand_gen;
  curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rand_gen, 1234ULL);
  curandGenerateUniform(rand_gen, random_numbers_, kNRandNumbers_);
  curandDestroyGenerator(rand_gen);
}

CudaInterfaceMultilayer::~CudaInterfaceMultilayer() {
  cudaFree(input_image_);
  cudaFree(result_image_);
  cudaFree(intermediate_image_arr_computing_);
  cudaFree(random_numbers_);
  cudaStreamDestroy(cuda_stream_);
}

void CudaInterfaceMultilayer::AddNewImage(
    const mini_tof::MiniTofImages &images) {
  for (int i = 0; i < images.size(); i++) {
    frequencies_[i] = (float)images[i].frequency;
    float coefficient_phase =
        1 / mini_tof::kSpeedOfLight * frequencies_[i] * 4 * kPi / 1000;

    for (int n = 0; n < kNPixels_; n++) {
      if (!images[i].exception[n]) {
        input_image_[2 * i * kNPixels_ + n] =
            (float)images[i].distance[n] * coefficient_phase;
        input_image_[2 * i * kNPixels_ + n + kNPixels_] =
            (float)images[i].amplitude[n] / 3000;
      } else {
        input_image_[2 * i * kNPixels_ + n] = (float)images[i].distance[n];
        input_image_[2 * i * kNPixels_ + n + kNPixels_] =
            (float)images[i].amplitude[n];
      }
    }
  }
  timestamp_ = images[0].timestamp;
  temperature_ = images[0].temperature;
}

mini_tof::MiniTofImages CudaInterfaceMultilayer::SolveMultilayerProblem(
    int min_amplitude) {
  auto start_time = std::chrono::high_resolution_clock::now();

  PsoWrapper(input_image_, intermediate_image_arr_computing_, result_image_,
             frequencies_, random_numbers_, cuda_stream_, kNParticles_,
             n_iterations_pso_, optimality_tolerance_gradient_decent_);

  cudaStreamAttachMemAsync(cuda_stream_, result_image_, 0, cudaMemAttachHost);

  cudaDeviceSynchronize();

  mini_tof::MiniTofImages images_out;
  images_out.resize(2);

  for (int i = 0; i < 2; i++) {
    images_out[i].distance.resize(kNPixels_);
    images_out[i].amplitude.resize(kNPixels_);
    images_out[i].exception.resize(kNPixels_);
    images_out[i].acquisition_mode = mini_tof::Mode::kDistanceAmplitudeMlt;
    images_out[i].temperature = temperature_;
    images_out[i].timestamp = timestamp_;
    images_out[i].frequency = 0;
  }
  for (int n = 0; n < kNPixels_; n++) {
    auto exception = result_image_[n] >= MIN_INVALID_VALUE;
    if (!exception) {
      auto d0 = result_image_[n] * 1000;
      auto d1 = result_image_[n + 2 * kNPixels_] * 1000;
      if (d1 > d0) {
        images_out[0].distance[n] = (uint16_t)d0;  // m to mm
        images_out[0].amplitude[n] =
            (uint16_t)(result_image_[n + kNPixels_] * 3000);
        images_out[1].distance[n] = (uint16_t)d1;  // m to mm
        images_out[1].amplitude[n] =
            (uint16_t)(result_image_[n + 3 * kNPixels_] * 3000);
      } else {
        images_out[0].distance[n] = (uint16_t)d1;  // m to mm
        images_out[0].amplitude[n] =
            (uint16_t)(result_image_[n + 3 * kNPixels_] * 3000);
        images_out[1].distance[n] = (uint16_t)d0;  // m to mm
        images_out[1].amplitude[n] =
            (uint16_t)(result_image_[n + kNPixels_] * 3000);
      }

      // remove low amplitude
      if (images_out[0].amplitude[n] < min_amplitude) {
        images_out[0].distance[n] = 0;
        images_out[0].amplitude[n] = 0;
      }
      if (images_out[1].amplitude[n] < min_amplitude) {
        images_out[1].distance[n] = 0;
        images_out[1].amplitude[n] = 0;
      }

      //sum up two distances if they are roughly the same
      if (abs(images_out[0].distance[n] - images_out[1].distance[n]) <
          kMinDistDifference) {
        images_out[0].distance[n] =
            (images_out[0].distance[n] + images_out[1].distance[n]) / 2;
        images_out[0].amplitude[n] =
            images_out[0].amplitude[n] + images_out[1].amplitude[n];
        images_out[1].distance[n] = 0;
        images_out[1].amplitude[n] = 0;
      }

    } else {
      images_out[0].distance[n] = (uint16_t)result_image_[kNPixels_ + n];
      images_out[0].amplitude[n] = (uint16_t)result_image_[1 * kNPixels_ + n];
      images_out[1].distance[n] = (uint16_t)result_image_[2 * kNPixels_ + n];
      images_out[1].amplitude[n] = (uint16_t)result_image_[3 * kNPixels_ + n];
      images_out[0].exception[n] = true;
      images_out[1].exception[n] = true;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time)
                      .count();

  std::cout << "Solved multilayer problem in " << duration << " milliseconds"
            << std::endl;

  return images_out;
}

}  // namespace mlt