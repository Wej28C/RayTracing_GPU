#ifndef CAMERA_H
#define CAMERA_H

#include "vecteur3d.h"
#include "point3d.h"
#include "ray.h"
struct Camera {
    point3d origin;
    vecteur3d lower_left_corner;
    vecteur3d horizontal;
    vecteur3d vertical;

    __host__ __device__ Camera() {
        // Configuration par défaut (adaptable)
        lower_left_corner = vecteur3d(-2.0f, -1.5f, -1.0f);
        horizontal = vecteur3d(4.0f, 0.0f, 0.0f);
        vertical = vecteur3d(0.0f, 3.0f, 0.0f);
        origin = point3d(0.0f, 0.0f, 0.0f);
    }

    __device__ Ray get_ray(float u, float v) const {
        vecteur3d dir = lower_left_corner + horizontal * u + vertical * v;
        return Ray(origin, dir);
    }
};

#endif