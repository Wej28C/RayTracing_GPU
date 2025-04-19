#include "include/camera.h"
#include "include/scene_gpu.h"
#include "include/cuda_utils.h"
#include <cuda_runtime.h>

int main() {
    // Configuration de l'image
    const int width = 800, height = 450, samples = 100;
    
    // ------------------------------------------
    // Étape 1 : Construction de la scène sur CPU
    // ------------------------------------------
    std::vector<Object_GPU> h_objects;
    std::vector<Material> h_materials;
    
    // Matériau Lambertian (sol)
    h_materials.push_back(Material{LAMBERTIAN, Color(0.8f, 0.8f, 0.0f)});
    
    // Sphère centrale
    h_objects.push_back(Object_GPU{
        .type = SPHERE,
        .sphere = {.center = point3d(0, 0, -2), .radius = 0.5f},
        .material_id = 0  // Index du matériau ci-dessus
    });
    
    // ... Ajouter d'autres objets

    // ------------------------------------------
    // Étape 2 : Transfert des données vers le GPU
    // ------------------------------------------
    Object_GPU* d_objects;
    Material* d_materials;
    
    // Allocation et copie des objets
    cudaMalloc(&d_objects, h_objects.size() * sizeof(Object_GPU));
    cudaMemcpy(d_objects, h_objects.data(), 
               h_objects.size() * sizeof(Object_GPU), 
               cudaMemcpyHostToDevice);
    
    // Allocation et copie des matériaux
    cudaMalloc(&d_materials, h_materials.size() * sizeof(Material));
    cudaMemcpy(d_materials, h_materials.data(), 
               h_materials.size() * sizeof(Material), 
               cudaMemcpyHostToDevice);
    
    // Structure de scène GPU
    SceneData d_scene = {
        .objects = d_objects,
        .materials = d_materials,
        .num_objects = static_cast<int>(h_objects.size()),
        .num_materials = static_cast<int>(h_materials.size())
    };

    // ------------------------------------------
    // Étape 3 : Initialisation des états aléatoires
    // ------------------------------------------
    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, width * height * sizeof(curandState));
    
    // Configuration des threads pour l'initialisation
    dim3 initBlocks(256);  // 256 threads par bloc
    dim3 initGrid((width * height + 255) / 256);  // Nombre de blocs nécessaires
    setup_rand_states<<<initGrid, initBlocks>>>(d_rand_states, time(nullptr));

    // ------------------------------------------
    // Étape 4 : Exécution du kernel de rendu
    // ------------------------------------------
    Color* d_image;
    cudaMalloc(&d_image, width * height * sizeof(Color));
    
    // Configuration des blocs (16x16 threads par bloc)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Appel du kernel
    render_kernel<<<gridSize, blockSize>>>(d_image, d_scene, width, height, samples, d_rand_states);

    // ------------------------------------------
    // Étape 5 : Récupération et sauvegarde de l'image
    // ------------------------------------------
    Color* h_image = new Color[width * height];
    cudaMemcpy(h_image, d_image, 
               width * height * sizeof(Color), 
               cudaMemcpyDeviceToHost);
    
    write_ppm("output_gpu.ppm", h_image, width, height);

    // ------------------------------------------
    // Étape 6 : Nettoyage de la mémoire
    // ------------------------------------------
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_rand_states);
    cudaFree(d_image);
    delete[] h_image;
    
    return 0;
}