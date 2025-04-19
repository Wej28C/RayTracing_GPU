#include "include/color.h"
#include <fstream>
#include <cstdlib>
/**
 * Sauvegarde le buffer de pixels dans un fichier PPM.
 * Format texte simple, compatible avec la plupart des visionneuses.
 */
void write_ppm(
    const std::string& filename, 
    Color* pixels, 
    int width, 
    int height
) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Erreur : impossible de créer " << filename << std::endl;
        return;
    }

    // En-tête PPM
    out << "P3\n" << width << " " << height << "\n255\n";

    // Écriture des pixels (lignes inversées pour le format PPM)
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            uint8_t r, g, b;
            pixels[j * width + i].to_uint8(r, g, b);  // Conversion [0, 255]
            out << static_cast<int>(r) << " "
                << static_cast<int>(g) << " "
                << static_cast<int>(b) << "\n";
        }
    }

    out.close();
}