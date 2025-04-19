#ifndef DEVICE_FUNCTIONS_H
#define DEVICE_FUNCTIONS_H

#include "scene_gpu.h"
#include "objet_gpu.h"
#include "material_gpu.h"

__device__ bool intersect_scene(const SceneData& scene, const Ray& ray, 
                              float t_min, float t_max, InfoIntersect& rec);

__device__ bool scatter_material(const Material& mat, const Ray& ray_in,
                               const InfoIntersect& rec, Color& attenuation,
                               Ray& scattered, curandStateXORWOW* rand_state);

__device__ bool intersect_object(const Object_GPU& obj, const Ray& ray,
                               float t_min, float t_max, InfoIntersect& rec);

#endif