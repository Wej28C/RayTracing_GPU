cmake_minimum_required(VERSION 3.10)
project(RayTracerSequential CXX)

# Configuration du compilateur C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fichiers sources
file(GLOB SOURCES "src/*.cpp" )

# Fichiers d'en-tête
include_directories(include)

# Création de l'exécutable
add_executable(raytracer_sequential ${SOURCES})

# Options de compilation (optimisations)
target_compile_options(raytracer_sequential PRIVATE -O3)