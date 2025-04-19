#include "cuda_utils.h"
#include <math.h>
// Génère une direction aléatoire sur la sphère unitaire (CUDA)

__device__ __noinline__ vecteur3d random_unit_vector(curandStateXORWOW* state) {
    float a = 2.0f * M_PI * curand_uniform(state);
    float z = 2.0f * curand_uniform(state) - 1.0f;
    float r = sqrtf(1.0f - z * z);
    return vecteur3d(r * cosf(a), r * sinf(a), z);
}


/*Initialisation des états aléatoires CURAND pour chaque thread.
 * Doit être appelé avant le kernel de rendu.
 */
__global__ void setup_rand_states(curandStateXORWOW* states, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

