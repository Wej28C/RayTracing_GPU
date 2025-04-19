#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <curand_kernel.h>
#include "vecteur3d.h"

// Déclaration seulement

__device__ vecteur3d random_unit_vector(curandStateXORWOW* state);

__global__ void setup_rand_states(curandStateXORWOW* states, unsigned int seed);

#endif