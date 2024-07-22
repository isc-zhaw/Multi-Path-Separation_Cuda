add_subdirectory(src)

set(LIBRARY_NAME multilayer_solver)

set(HEADER_LIST "include/cuda_interface_multilayer.h")

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE_ZHAW})

find_package(CUDAToolkit REQUIRED)

add_library(${LIBRARY_NAME} cuda_interface_multilayer.cpp ${HEADER_LIST})

target_include_directories(${LIBRARY_NAME} PUBLIC include)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(${LIBRARY_NAME} PROPERTIES LANGUAGE CUDA)
set_target_properties(${LIBRARY_NAME} PROPERTIES LANGUAGE CXX)

set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON) #Set TARGET Property for cuda
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)  # this is required for some reason

target_link_libraries(${LIBRARY_NAME} PRIVATE CUDA::cudart)
target_link_libraries(${LIBRARY_NAME} PRIVATE CUDA::curand)
target_link_libraries(${LIBRARY_NAME} PRIVATE multilayer_solver_cuda)
target_link_libraries(${LIBRARY_NAME} PUBLIC mini_tof_typedef)

target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dlink>)