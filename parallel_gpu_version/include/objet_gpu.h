// ==== Fichier : object_gpu.h ====
#ifndef OBJECT_GPU_H
#define OBJECT_GPU_H

#include "InfoIntersect.h"
#include "vecteur3d.h"
#include "point3d.h"
#include "ray.h"
enum ObjectType { SPHERE, TRIANGLE, PLANE };

struct Object_GPU {
    ObjectType type;
    union {
        struct { point3d center; float radius; } sphere;
        struct { point3d v0; point3d v1;  point3d v2; } triangle;
        struct { point3d point; vecteur3d normal; } plane;
    };
    int material_id; // Index dans le tableau de matériaux GPU
};

// Fonction d'intersection générique (dispatch manuel)
/*
__device__ bool intersect_object(
    const Object_GPU& obj,
    const Ray& ray,
    float t_min,
    float t_max,
    InfoIntersect& rec
) {
    switch (obj.type) {
        case SPHERE: {
            vecteur3d oc = ray.origin() - obj.sphere.center;
            float a = dot(ray.direction(), ray.direction());
            float half_b = dot(oc, ray.direction());
            float c = dot(oc, oc) - obj.sphere.radius * obj.sphere.radius;
            float discriminant = half_b * half_b - a * c;

            if (discriminant < 0) return false;
            float sqrtd = sqrtf(discriminant);
            float root = (-half_b - sqrtd) / a;

            if (root < t_min || root > t_max) {
                root = (-half_b + sqrtd) / a;
                if (root < t_min || root > t_max) return false;
            }

            rec.t = root;
            rec.p = ray.emis(rec.t);
            vecteur3d outward_normal = (rec.p - obj.sphere.center) / obj.sphere.radius;
            rec.set_face_normal(ray, outward_normal);
            rec.material_id = obj.material_id;

            return true;
        }

        // Intersection pour les triangles (simplifiée)
        case TRIANGLE: {
            // Implémentation de l'algorithme Möller–Trumbore
            vecteur3d edge1 = obj.triangle.v1 - obj.triangle.v0;
            vecteur3d edge2 = obj.triangle.v2 - obj.triangle.v0;
            vecteur3d h = cross(ray.direction(), edge2);
            float a = dot(edge1, h);
            // ... (suite du code)
            return true;
        }

        // Intersection pour les plans
        case PLANE: {
            float denom = dot(obj.plane.normal, ray.direction());
            if (fabsf(denom) < 1e-6f) return false;
            float t = dot(obj.plane.point - ray.origin(), obj.plane.normal) / denom;
            if (t < t_min || t > t_max) return false;

            rec.t = t;
            rec.p = ray.emis(t);
            rec.set_face_normal(ray, obj.plane.normal);
            rec.material_id = obj.material_id;
            return true;
        }
        default: return false;
    }
    
}
*/
__device__ bool intersect_object(const Object_GPU& obj, const Ray& ray,
    float t_min, float t_max, InfoIntersect& rec);
#endif