<h2 align="center">Compact 3D Time-of-Flight Sensor with Real-Time Multi-Path Separation</h2>
  <p align="center">
  <strong>Matthias Ludwig</strong>,
  <strong>Roman Gubler</strong>,
  <strong>Samuel Schüller</strong>
  <strong>Teddy Loeliger</strong>,
  </p>
  <p align="center">
Institute of Signal Processing and Wireless Communications (ISC),<br>ZHAW Zurich University of Applied Sciences
</p>

The acquisition of 3D data using indirect 3D Time-of-Flight (3D ToF) sensors is a well-established technology. However, Multi-Path Interference (MPI) is a key issue and causes significant errors in the distance measurements. This paper presents a novel compact 3D ToF sensor with real-time multi-path separation. The sensor combines multiple measurements at different modulation frequencies and employs Particle Swarm Optimization (PSO) in combination with Sequential Quadratic Programming (SQP) to separate the different paths. The sensor is used to demonstrate the separation of two different paths in each pixel of the entire image in real-time. Evaluation in various laboratory and real-world scenarios reveals a significant improvement in distance accuracy compared to a standard 3D ToF sensor. Additionally, the proposed algorithm exhibits processing speeds that are orders of magnitude faster than previously reported separation algorithms with comparable accuracy.


## Setup
This code is not complete to be runnable. So no setup is described.

## Code structure
The folder `multilayer_solver` includes the file `cuda_interface_multilayer.cpp`. This file is the interface to the multi-path solver.

The object `CudaInterfaceMultilayer` must be initialized.
Through the function `AddNewImage()` new images can be added to the CUDA buffer. After the images are added and converted, the function `SolveMultilayerProblem()` can be called. These two functions should run consecutively and not in parallel.

The input and output of the `AddNewImage()` and `SolveMultilayerProblem()` should be changed to the used data format.



## Citation
```
@software{ludwig_real_time_mps,
author = {Ludwig, Matthias and Gubler, Roman and Schüller, Samuel and Loeliger, Teddy},
license = {MIT},
title = {{Compact 3D Time-of-Flight Sensor with Real-Time Multi-Path Separation}},
url = {https://github.com/isc-zhaw/Multi-Path-Separation_Cuda.git}
}
```