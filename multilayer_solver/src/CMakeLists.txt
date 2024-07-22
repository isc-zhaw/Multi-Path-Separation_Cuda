set(LIBRARY_NAME multilayer_solver_cuda)

set(HEADER_LIST "../include/private/cuda_pso.cuh"
        "../include/private/helper_sqp.cuh"
        "../include/private/sqp_cuda.cuh"
        "../include/private/sqp_cuda_types.cuh")


set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE_ZHAW})

add_library(${LIBRARY_NAME} cuda_pso.cu
        helper_sqp.cu
        sqp_cuda.cu
        ${HEADER_LIST})

target_include_directories(${LIBRARY_NAME} PUBLIC ../include)

set_target_properties(${LIBRARY_NAME} PROPERTIES LANGUAGE CUDA)
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON) #Set TARGET Property for cuda
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)  # this is required for some reason

set(CMAKE_CUDA_FLAGS "-use_fast_math")
find_package(CUDAToolkit REQUIRED)

target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dlink>)

