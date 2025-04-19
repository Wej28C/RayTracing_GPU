// Color.h
#ifndef COLOR_H
#define COLOR_H

#include "vecteur3d.h"

class Color : public vecteur3d {
public:
    Color() : vecteur3d() {}
    Color(float r, float g, float b) : vecteur3d(r, g, b) {}
    Color(const vecteur3d& v) : vecteur3d(v) {}
    float r() const { return c[0]; }
    float g() const { return c[1]; }
    float b() const { return c[2]; }
    
    // fixe la valeur de la couleur entre 0 et 1
    Color fixed() const {
        float red = std::max(0.0f, std::min(1.0f, c[0]));
        float green = std::max(0.0f, std::min(1.0f, c[1]));
        float blue = std::max(0.0f, std::min(1.0f, c[2]));
        return Color(red, green, blue);
    }
    
    // Convert to 8-bit color values (0-255)
    void to_uint8(uint8_t &r, uint8_t &g, uint8_t &b) const {
        Color fixed_color = fixed();
        r = static_cast<uint8_t>(255.999f * fixed_color.r());
        g = static_cast<uint8_t>(255.999f * fixed_color.g());
        b = static_cast<uint8_t>(255.999f * fixed_color.b());
    }
    Color operator*(const Color& other) const {
        return Color(c[0] * other.c[0], c[1] * other.c[1], c[2] * other.c[2]);
    }
};

#endif // COLOR_H