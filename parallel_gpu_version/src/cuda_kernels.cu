#include "device_functions.h"
/*
#include "objet_gpu.h"
#include "scene_gpu.h"*/
#include "cuda_utils.h"
#include "camera.h"
// Déclarations des fonctions utilisées

/*__device__ bool intersect_scene(const SceneData& scene, const Ray& ray, float t_min, float t_max, InfoIntersect& rec);
__device__ bool scatter_material(const Material& mat, const Ray& ray_in, const InfoIntersect& rec, Color& attenuation, Ray& scattered, curandStateXORWOW* rand_state);*/
/**
 * Calcule la couleur d'un rayon en prenant en compte les intersections et les matériaux.
 * Version itérative optimisée pour CUDA (pas de récursivité).
 */
__device__ Color ray_color(
    const Ray& r,
    const SceneData& scene,
    curandState* rand_state,
    int max_depth = 50
) {
    Ray current_ray = r;
    Color attenuation(1.0f);
    
    for (int depth = 0; depth < max_depth; depth++) {
        InfoIntersect rec;
        if (!intersect_scene(scene, current_ray, 0.001f, 1e6f, rec)) {
            // Fond dégradé
            float t = 0.5f * (current_ray.direction().y + 1.0f);
            return attenuation * ((1.0f - t) * Color(1.0f) + t * Color(0.5f, 0.7f, 1.0f));
        }

        Material mat = scene.materials[rec.material_id];
        Ray scattered;
        Color mat_attenuation;
         // Calcul de la diffusion (ex: Lambertian)
        if (!scatter_material(mat, current_ray, rec, mat_attenuation, scattered, rand_state)) {
            return Color(0.0f);
        }

        attenuation = attenuation * mat_attenuation;
        current_ray = scattered;    // Met à jour le rayon pour la prochaine itération
    }
    
    return Color(0.0f); // Atteint la profondeur max
}

/**
 * Kernel CUDA principal : calcule la couleur de chaque pixel en parallèle.
 * Un thread GPU est responsable d'un pixel.
 */
__global__ void render_kernel(
    Color* image,
    SceneData scene,
    Camera cam,
    int width,
    int height,
    int samples,
    curandState* rand_states
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    curandState local_state = rand_states[y * width + x];
    Color pixel_color(0.0f);
    // Échantillonnage anti-aliasing
    for (int s = 0; s < samples; s++) {
        float u = (x + curand_uniform(&local_state)) / (width - 1);
        float v = (y + curand_uniform(&local_state)) / (height - 1);
        Ray ray = cam.get_ray(u, v);
        pixel_color += ray_color(ray, scene, &local_state);
    }
      // Moyenne des échantillons et sauvegarde
    image[y * width + x] = pixel_color / samples;
    rand_states[y * width + x] = local_state; // Sauvegarde de l'état
}