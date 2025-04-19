#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <curand_kernel.h>

// Génère une direction aléatoire sur la sphère unitaire (CUDA)
__device__ vecteur3d random_unit_vector(curandState* state) {
    float theta = 2.0f * M_PI * curand_uniform(state);
    float phi = acos(2.0f * curand_uniform(state) - 1.0f);
    return vecteur3d(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi));
}

/*Initialisation des états aléatoires CURAND pour chaque thread.
 * Doit être appelé avant le kernel de rendu.
 */
__global__ void setup_rand_states(curandState* states, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

#endif