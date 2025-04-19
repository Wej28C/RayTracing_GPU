#ifndef CAMERA_H
#define CAMERA_H

#include "vecteur3d.h"

struct Camera {
    vecteur3d origin, lower_left_corner, horizontal, vertical;

    __host__ __device__ Camera() {
        // Configuration par défaut (adaptable)
        lower_left_corner = vecteur3d(-2.0f, -1.5f, -1.0f);
        horizontal = vecteur3d(4.0f, 0.0f, 0.0f);
        vertical = vecteur3d(0.0f, 3.0f, 0.0f);
        origin = vecteur3d(0.0f, 0.0f, 0.0f);
    }

    __device__ Ray get_ray(float u, float v) const {
        return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }
};

#endif