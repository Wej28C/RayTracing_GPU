#ifndef PPMEXPORTER_H
#define PPMEXPORTER_H

#include <vector>
#include <fstream>

struct Color {
    float r, g, b;
};

class PPMExporter {
public:
    static void exportToPPM(const std::vector<Color>& pixels, int width, int height, const std::string& filename) {
        std::ofstream file(filename);
        file << "P3\n" << width << " " << height << "\n255\n";
        for (const auto& pixel : pixels) {
            file << static_cast<int>(pixel.r * 255) << " "
                 << static_cast<int>(pixel.g * 255) << " "
                 << static_cast<int>(pixel.b * 255) << "\n";
        }
    }
};

#endif // PPMEXPORTER_H