#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "scene.h"
#include "Triangle.h"
#include "plane.h"
// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;


// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

inline float random_float() {
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_float();
}
/*
float random_float() {
    static bool initialized = false;
    if (!initialized) {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        initialized = true;
    }
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}*/
// Fonction pour écrire l'image au format PPM
void write_ppm(const std::string&  filename, Color* pixels, int width, int height) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    out << "P3\n" << width << " " << height << "\n255\n";
    
    for (int j = height-1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            uint8_t r, g, b;
            pixels[j*width + i].to_uint8(r, g, b);
            out << static_cast<int>(r) << " "
                << static_cast<int>(g) << " "
                << static_cast<int>(b) << "\n";
        }
    }
    
    out.close();
}

// Fonction pour calculer la couleur d'un rayon
Color ray_color(const Ray& r, const Scene& scene, int depth) {
    InfoIntersect rec;
    
    // Si on a dépassé la profondeur maximale, retourner noir
    if (depth <= 0)
        return Color(0, 0, 0);
    
    if (scene.intersectScene(r, 0.001f, std::numeric_limits<float>::infinity(), rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.material->scatter(r, rec, attenuation, scattered)){
            Color scattered_color = ray_color(scattered, scene, depth-1);
            // Multiplication composante par composante
            return Color(
                attenuation.r() * scattered_color.r(),
                attenuation.g() * scattered_color.g(),
                attenuation.b() * scattered_color.b()
            );
        }
        return Color(0, 0, 0);
    }
    
    // Fond dégradé bleu-blanc si aucun objet n'est touché
    vecteur3d unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.getY() + 1.0f);
    return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Dimensions de l'image
    const float aspect_ratio = 16.0f / 9.0f;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;
    
    // Création de la scène
    Scene scene;
    /*
    // Ajout de matériaux
    auto material_ground = std::make_shared<lambertian>(Color(0.8f, 0.8f, 0.0f));
    auto material_center = std::make_shared<lambertian>(Color(0.7f, 0.3f, 0.3f));
    auto material_left   = std::make_shared<lambertian>(Color(0.8f, 0.8f, 0.8f));
    auto material_right  = std::make_shared<lambertian>(Color(0.8f, 0.6f, 0.2f));
    auto material_plane = std::make_shared<lambertian>(Color(0.2f, 0.6f, 0.2f)); // Vert
    auto material_triangle = std::make_shared<lambertian>(Color(0.8f, 0.3f, 0.3f)); // Rouge
    // Ajout de sphères à la scène
   // scene.ajouterObjet(std::make_shared<Sphere>(point3d( 0.0f, -100.5f, -1.0f), 100.0f, material_ground));
      scene.ajouterObjet(std::make_shared<Sphere>(point3d( 0.0f,    0.0f, -1.0f),   0.5f, material_center));
   // scene.ajouterObjet(std::make_shared<Sphere>(point3d(-1.0f,    0.0f, -1.0f),   0.5f, material_left));
   // scene.ajouterObjet(std::make_shared<Sphere>(point3d( 1.0f,    0.0f, -1.0f),   0.5f, material_right));
     // Ajout d'un PLAN (sol)
     point3d plane_point(0.0f, -0.5f, 0.0f); // Position du plan
     vecteur3d plane_normal(0.0f, 1.0f, 0.0f); // Normale vers le haut
     scene.ajouterObjet(std::make_shared<Plane>(plane_point, plane_normal, material_plane));
 
     // Ajout d'un TRIANGLE
     point3d tri_p1(-1.0f, 0.0f, -2.0f); // Point 1 du triangle
     point3d tri_p2(1.0f, 0.0f, -2.0f);  // Point 2
     point3d tri_p3(0.0f, 1.0f, -2.0f);  // Point 3
     scene.ajouterObjet(std::make_shared<Triangle>(tri_p1, tri_p2, tri_p3, material_triangle));
        */
    auto mat_ground = std::make_shared<lambertian>(Color(0.8f, 0.8f, 0.0f)); // Jaune
    auto mat_sphere = std::make_shared<lambertian>(Color(0.7f, 0.3f, 0.3f)); // Rouge
    auto mat_triangle = std::make_shared<lambertian>(Color(0.2f, 0.5f, 0.8f)); // Bleu

    // Ajout des objets (position ajustée)
    scene.ajouterObjet(std::make_shared<Plane>(
        point3d(0, -1, 0),  // Position
        vecteur3d(0, 1, 0),  // Normale vers le haut
        mat_ground
    ));

    scene.ajouterObjet(std::make_shared<Sphere>(
        point3d(0, 0, -2),   // Centre
        0.5f,                // Rayon
        mat_sphere
    ));

    scene.ajouterObjet(std::make_shared<Triangle>(
        point3d(-1, 0, -1),  // Point A
        point3d(1, 0, -1),   // Point B
        point3d(0, 2, -1),   // Point C
        mat_triangle
    ));
     // Création de la caméra
    Camera cam;
    
    // Allocation du buffer d'image
    Color* image = new Color[image_width * image_height];
    
    // Rendu de l'image
    // c'est cette partie que williamm gpu se 
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            Color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = (i + random_float()) / (image_width-1);
                float v = (j + random_float()) / (image_height-1);
                Ray r = cam.getRay(u, v);
                pixel_color += ray_color(r, scene, max_depth);
            }
            pixel_color /= samples_per_pixel;
            image[j*image_width + i] = pixel_color;
        }
    }
    
    // Écriture de l'image
    write_ppm("output.ppm", image, image_width, image_height);
    
    // Nettoyage
    delete[] image;
    
    std::cerr << "\nDone.\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\nTemps d'exécution : " << duration.count() << " secondes\n";
    return 0;
}
