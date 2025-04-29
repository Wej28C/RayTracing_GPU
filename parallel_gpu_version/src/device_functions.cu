#include "device_functions.h"
#include "cuda_utils.h"

// Implémentation unique de intersect_scene

__device__ bool intersect_scene(
    const SceneData& scene,
    const Ray& ray,
    float t_min,
    float t_max,
    InfoIntersect& rec
) {
    InfoIntersect temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < scene.num_objects; i++) {
        if (intersect_object(scene.objects[i], ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// Implémentation unique de scatter_material

__device__ bool scatter_material(
    const Material& mat,
    const Ray& ray_in,
    const InfoIntersect& rec,
    Color& attenuation,
    Ray& scattered,
    curandStateXORWOW* rand_state
) {
    switch (mat.type) {
        case LAMBERTIAN: {
            vecteur3d scatter_dir = rec.normal + random_unit_vector(rand_state);
            scattered = Ray(rec.p, scatter_dir);
            attenuation = mat.albedo;
            return true;
        }
        
        default: return false;
    }
}

// Implémentation unique de intersect_object
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
           /*/ vecteur3d edge1 = obj.triangle.v1 - obj.triangle.v0;
            vecteur3d edge2 = obj.triangle.v2 - obj.triangle.v0;
            vecteur3d h = cross(ray.direction(), edge2);
            float a = dot(edge1, h);*/
                vecteur3d edge1 = obj.triangle.v1 - obj.triangle.v0;
                vecteur3d edge2 = obj.triangle.v2 - obj.triangle.v0;
                vecteur3d h = cross(ray.direction(), edge2);
                float a = dot(edge1, h);

                if (fabsf(a) < 1e-8f) return false;  // Le rayon est parallèle au triangle

                float f = 1.0f / a;
                vecteur3d s = ray.origin() - obj.triangle.v0;
                float u = f * dot(s, h);
                if (u < 0.0f || u > 1.0f) return false;

                vecteur3d q = cross(s, edge1);
                float v = f * dot(ray.direction(), q);
                if (v < 0.0f || u + v > 1.0f) return false;

                float t = f * dot(edge2, q);
                if (t < t_min || t > t_max) return false;

                rec.t = t;
                rec.p = ray.emis(t);
                vecteur3d outward_normal =cross(edge1, edge2).normalized();
                rec.set_face_normal(ray, outward_normal);
                rec.material_id = obj.material_id;

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