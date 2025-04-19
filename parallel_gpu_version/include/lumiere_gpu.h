#ifndef LUMIERE_GPU_H
#define LUMIERE_GPU_H

#include "point3d.h"
#include "color.h"

// Structure pour les lumières (CUDA-friendly)
struct Lumiere_GPU {
    point3d position;
    Color color;

    __host__ __device__ Lumiere_GPU() 
        : position(point3d(0,0,0)), color(Color(1,1,1)) {}

    __host__ __device__ Lumiere_GPU(const point3d& pos, const Color& col)
        : position(pos), color(col) {}
};

#endif