#include "ppm_writer.h"
#include <fstream>
#include <iostream>

void write_ppm(const std::string& filename, const Color* pixels, int width, int height) {
    std::ofstream out(filename);
    out << "P3\n" << width << " " << height << "\n255\n";
    
    for (int y = height-1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            const Color& pixel = pixels[y * width + x];
            int r = static_cast<int>(255.999f * pixel.r);
            int g = static_cast<int>(255.999f * pixel.g);
            int b = static_cast<int>(255.999f * pixel.b);
            out << r << " " << g << " " << b << "\n";
        }
    }
}