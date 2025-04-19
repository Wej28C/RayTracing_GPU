#ifndef VECTEUR3D_H
#define VECTEUR3D_H

#include <cmath>

// Structure légère pour CUDA (pas d'héritage)
struct vecteur3d {
    float x, y, z;

    // Constructeurs pour CPU/GPU
    __host__ __device__ vecteur3d() : x(0), y(0), z(0) {}
    __host__ __device__ vecteur3d(float x, float y, float z) : x(x), y(y), z(z) {}

    // Opérations de base (CUDA-friendly)
    __host__ __device__ vecteur3d operator-() const { 
        return vecteur3d(-x, -y, -z); 
    }

    __host__ __device__ vecteur3d& operator+=(const vecteur3d& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ static float dot(const vecteur3d& u, const vecteur3d& v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    }
};

// Fonctions externes (pour éviter les conflits CUDA)
__host__ __device__ inline vecteur3d operator+(const vecteur3d& a, const vecteur3d& b) {
    return vecteur3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline vecteur3d operator*(float t, const vecteur3d& v) {
    return vecteur3d(t*v.x, t*v.y, t*v.z);
}

#endif