cmake_minimum_required(VERSION 3.18)
project(RayTracerCUDA LANGUAGES CXX CUDA)

# Configuration des standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configuration CUDA spécifique
set(CMAKE_CUDA_ARCHITECTURES "80") # Détection automatique de l'architecture
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # Essentiel pour le device code

# Options de compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3 --use_fast_math -rdc=true")

# Inclusions
include_directories(include)

# Fichiers sources (explicite plutôt que GLOB)
set(SOURCES
    src/main.cu
    src/cuda_kernels.cu
    src/cuda_utils.cu
    src/device_functions.cu
    src/ppm_writer.cpp
)

# Création de l'exécutable
add_executable(raytracer_cuda ${SOURCES})

# Détection des librairies CUDA
find_package(CUDAToolkit REQUIRED)

# Liaisons
target_link_libraries(raytracer_cuda 
    PRIVATE 
    CUDA::cudart
    CUDA::curand
)

# Configuration spécifique pour les fichiers CUDA
set_source_files_properties(
    src/cuda_kernels.cu
    src/cuda_utils.cu
    PROPERTIES
    LANGUAGE CUDA
)

# Vérification
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")