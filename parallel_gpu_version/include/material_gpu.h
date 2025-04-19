// ==== Fichier : gpu_material.h ====
#ifndef GPU_MATERIAL_H
#define GPU_MATERIAL_H

#include "vecteur3d.h"
#include "InfoIntersect.h"
#include "ray.h"
#include "color.h"
#include"cuda_utils.h"
#include <curand_kernel.h> 
enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC };

struct Material {
    MaterialType type;
    Color albedo;
   // float fuzz;  // Pour METAL
    //float ir;    // Pour DIELECTRIC
};

// Fonction de scattering générique (dispatch manuel)
/*
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
    */
    __device__ bool scatter_material(const Material& mat, const Ray& ray_in,
        const InfoIntersect& rec, Color& attenuation,
        Ray& scattered, curandStateXORWOW* rand_state) ; 

#endif