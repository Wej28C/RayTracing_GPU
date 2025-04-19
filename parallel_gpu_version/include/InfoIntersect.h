// ==== Fichier : gpu_info_intersect.h ====
#ifndef INFO_INTERSECT_H
#define INFO_INTERSECT_H

#include "vecteur3d.h"
#include "point3d.h"
#include "ray.h"
struct InfoIntersect {
    point3d p;
    vecteur3d normal;
    int material_id; // Index dans le tableau de matériaux GPU
    float t;
    bool front_face;

    __device__ void set_face_normal(const Ray& ray, const vecteur3d& outward_normal) {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -1.0f * outward_normal;
    }
};

#endif