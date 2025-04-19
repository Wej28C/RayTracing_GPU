#ifndef POINT3D_H
#define POINT3D_H

#include "vecteur3d.h"

// Structure pour les points (compatible GPU)
struct point3d {
    float x, y, z;

    __host__ __device__ point3d() : x(0), y(0), z(0) {}
    __host__ __device__ point3d(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ vecteur3d operator-(const point3d& p) const {
        return vecteur3d(x - p.x, y - p.y, z - p.z);
    }

    __host__ __device__ point3d operator+(const vecteur3d& v) const {
        return point3d(x + v.x, y + v.y, z + v.z);
    }
};

#endif