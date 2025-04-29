#include "camera.h"
#include "scene_gpu.h"
#include "cuda_utils.h"
#include "device_functions.h"
#include <cuda_runtime.h>
#include <vector> 
#include "ppm_writer.h"  
#include <chrono>
#include <iostream>

// Déclaration du kernel
__global__ void render_kernel(Color* image, SceneData scene, Camera cam, int width, int height, int samples, curandState* rand_states);
int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    // Configuration de l'image
    const int width = 800, height = 450, samples = 100;
    
    // ------------------------------------------
    // Étape 1 : Construction de la scène sur CPU
    // ------------------------------------------
    std::vector<Object_GPU> h_objects;
    std::vector<Material> h_materials;
    /*
    // Matériau Lambertian (sol)
    h_materials.push_back(Material{LAMBERTIAN, Color(0.8f, 0.8f, 0.0f)});
    
    // Sphère centrale
    h_objects.push_back(Object_GPU{
        .type = SPHERE,
        .sphere = {.center = point3d(0, 0, -2), .radius = 0.5f},
        .material_id = 0  // Index du matériau ci-dessus
    });
    
    // ... Ajouter d'autres objets
*/
//
/*
//----------------TEST1------------------//
        // Matériau Lambertian (sol)
   h_materials.push_back(Material{LAMBERTIAN, Color(0.8f, 0.8f, 0.0f)});
   
   // Sphère centrale
   h_objects.push_back(Object_GPU{
       .type = SPHERE,
       .sphere = {.center = point3d(0, 0, -2), .radius = 0.5f},
       .material_id = 0  // Index du matériau ci-dessus
   });
/*
   //----------FIN TEST1------------------//
   
   //----------------TEST2------------------//
   h_materials.push_back(Material{LAMBERTIAN, Color(0.8f, 0.8f, 0.0f)}); // sol jaune
   h_materials.push_back(Material{LAMBERTIAN, Color(0.2f, 0.3f, 0.7f)}); // sphère Bleu 

   // Sol plan 
   h_objects.push_back(Object_GPU{
       .type = PLANE,
       .plane = {.point = point3d(0, -0.5f, -1), .normal= vecteur3d(0,1,0)},
       .material_id = 0
   });

   // Sphère flottante
   h_objects.push_back(Object_GPU{
       .type = SPHERE,
       .sphere = {.center = point3d(0, 0, -1.5f), .radius = 0.5f},
       .material_id = 1
   });
    //----------FIN TEST2------------------//
    */
   /*
    //----------------TEST3------------------//
    h_materials.push_back(Material{LAMBERTIAN, Color(0.8f, 0.5f, 0.2f)}); // Triangle orange
    h_materials.push_back(Material{LAMBERTIAN, Color(0.1f, 0.8f, 0.1f)}); // sphere vert 
 
    // Sphère
    h_objects.push_back(Object_GPU{
        .type = SPHERE,
        .sphere = {.center = point3d(0, 0, -1.5f), .radius = 0.5f},
        .material_id = 1
    });
 
    // Triangle
    h_objects.push_back(Object_GPU{
        .type = TRIANGLE,
        .triangle = {
            .v0 = point3d(-1.5f, -0.5f, -3.0f),
            .v1 = point3d(1.5f, -0.5f, -3.0f),
            .v2 = point3d(0.0f, 1.5f, -3.0f)
        },
        .material_id = 0
    });
  */  //----------FIN TEST3------------------//
    
    //----------------TEST4------------------//
    // // 3 spheres avec mat et tailles différentes
    h_materials.push_back(Material{LAMBERTIAN, Color(0.9f, 0.1f, 0.1f)}); // rouge
    h_materials.push_back(Material{LAMBERTIAN, Color(0.1f, 0.9f, 0.1f)}); // vert 
    h_materials.push_back(Material{LAMBERTIAN, Color(0.1f, 0.1f, 0.9f)}); // bleu 
    h_materials.push_back(Material{LAMBERTIAN, Color(0.6f, 0.6f, 0.6f)}); // sol 
 
    h_objects.push_back(Object_GPU{.type=SPHERE, .sphere={.center = point3d(-1.0f, 0.0f, -2.5f), .radius = 0.4f}, 0});
    h_objects.push_back(Object_GPU{.type=SPHERE, .sphere={.center = point3d(0.0f, 0, -1.5f), .radius = 0.5f}, 1});
    h_objects.push_back(Object_GPU{.type=SPHERE, .sphere={.center = point3d(1.0f, 0, -2.0f), .radius = 0.3f}, 2});
   h_objects.push_back(Object_GPU{
    .type = PLANE,
    .plane = {.point = point3d(0, -0.5f, 0), .normal= vecteur3d(0,1,0)},
    .material_id = 3
    });
    //----------FIN TEST4------------------//
   /* //----------------TEST5------------------//
    h_materials.push_back(Material{LAMBERTIAN, Color(0.8f, 0.8f, 0.8f)}); // sol
    h_materials.push_back(Material{LAMBERTIAN, Color(0.9f, 0.1f, 0.1f)}); // sphère1 RED
    h_materials.push_back(Material{LAMBERTIAN, Color(0.2f, 0.3f, 0.8f)}); // sphère2 BLEU
    h_materials.push_back(Material{LAMBERTIAN, Color(0.1f, 0.9f, 0.6f)}); // triangle VERT TURQUOISE
 
    h_objects.push_back(Object_GPU{
        .type = PLANE,
        .plane = {.point = point3d(0, -1.0f, 0), .normal= vecteur3d(0,1,0)},
        .material_id = 0
        });
    h_objects.push_back(Object_GPU{.type=SPHERE, .sphere={.center = point3d(-0.75f, -0.5f, -1.8f), .radius = 0.5f}, 1});
    h_objects.push_back(Object_GPU{.type=SPHERE, .sphere={.center = point3d(0.75f, -0.5f, -1.2f), .radius = 0.5f}, 2});
    h_objects.push_back(Object_GPU{
        .type=TRIANGLE, .triangle={.v0 = point3d(-1, 0.0, -3), .v1 = point3d(1, 0.0, -3), .v2 = point3d(0, 2, -3)}, 3});
    //----------FIN TEST5------------------/
*/
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
    /* timer*/
   
    Color* d_image;
    cudaMalloc(&d_image, width * height * sizeof(Color));

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    // Configuration des blocs (16x16 threads par bloc)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    Camera cam;
   
  
    
    cudaEventRecord(start_gpu);
    // Appel du kernel
    render_kernel<<<gridSize, blockSize>>>(d_image, d_scene, cam, width, height, samples, d_rand_states);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);
    // ------------------------------------------
    // Étape 5 : Récupération et sauvegarde de l'image
    // ------------------------------------------
    Color* h_image = new Color[width * height];
    cudaMemcpy(h_image, d_image, 
               width * height * sizeof(Color), 
               cudaMemcpyDeviceToHost);
    
   // write_ppm("output_gpu.ppm", h_image, width, height);
 //   std::string filename = "output_test" + std::to_string(test_id) + ".ppm";
    write_ppm("output_gpu.ppm", h_image, width, height);
    
    // ------------------------------------------
    // Étape 6 : Nettoyage de la mémoire
    // ------------------------------------------
    cudaFree(d_objects);
    cudaFree(d_materials);
    cudaFree(d_rand_states);
    cudaFree(d_image);
    delete[] h_image;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Affichage des résultats
    std::cout << "=== Performance ===" << std::endl;
    std::cout << "Temps total d'exécution: " << duration << " ms" << std::endl;
    std::cout << "Temps GPU (kernel seulement): " << gpu_ms << " ms" << std::endl;
    std::cout << "===================" << std::endl;

    // Nettoyage des événements CUDA
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    return 0;
}