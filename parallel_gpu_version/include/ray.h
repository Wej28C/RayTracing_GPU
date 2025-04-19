#ifndef RAY_H
#define RAY_H

#include "vecteur3d.h"
#include "point3d.h"

struct Ray {

    point3d orig;
    vecteur3d dir;
 
    __host__ __device__ Ray() {}

    __host__ __device__ Ray(const point3d& origin, const vecteur3d& direction) : orig(origin), dir(direction) {}

    __host__ __device__ const point3d& origin() const  { return orig; }
    __host__ __device__ const vecteur3d& direction() const { return dir; }

    __host__ __device__ point3d emis(float t) const {
        return orig + t*dir;
    }
};

#endif