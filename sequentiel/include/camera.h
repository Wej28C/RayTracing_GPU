#ifndef CAMERA_H
#define CAMERA_H
#include "ray.h"

class Camera {
    public: 
    point3d origin;
    vecteur3d lower_left_corner;
    vecteur3d horizontal;
    vecteur3d vertical;

    Camera() {
        origin = point3d(0, 0, 0);
        lower_left_corner = vecteur3d(-2.0, -1.5, -1.0);
        horizontal = vecteur3d(4.0, 0.0, 0.0);
        vertical = vecteur3d(0.0, 3.0, 0.0);
    }

    Ray getRay(float u, float v) const {
        vecteur3d dir = lower_left_corner + horizontal * u + vertical * v;
        return Ray(origin, dir);
    }
   
}; 
#endif // CAMERA_H