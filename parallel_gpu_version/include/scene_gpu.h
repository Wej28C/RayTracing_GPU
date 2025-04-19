// ==== Fichier : gpu_scene.h ====
#ifndef SCENE_GPU_H
#define SCENE_GPU_H

#include "objet_gpu.h"
#include "material_gpu.h"
#include "lumiere_gpu.h"
struct SceneData {
    Object_GPU* objects;    // Tableau d'objets
    Material* materials;    // Tableau de matériaux
    Lumiere_GPU* lights; // Tableau de lumières
    int num_lights;        // Nombre de lumières
    int num_objects;
    int num_materials;
};

// Fonction d'intersection de la scène
/*
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
*/
// Déclaration seulement (pas d'implémentation)
__device__ bool intersect_scene(const SceneData& scene, const Ray& ray, 
    float t_min, float t_max, InfoIntersect& rec);
#endif