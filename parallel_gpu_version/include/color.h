#ifndef COLOR_H
#define COLOR_H
#include <cstdint>
#include "vecteur3d.h"
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
// Structure indépendante (pas d'héritage)
struct Color {
    float r, g, b;

    CUDA_CALLABLE Color() : r(0), g(0), b(0) {}
    CUDA_CALLABLE Color(float r, float g, float b) : r(r), g(g), b(b) {}
    CUDA_CALLABLE Color(float value) : r(value), g(value), b(value) {}
   // CUDA_CALLABLE Color(float r, float g, float b) : vecteur3d(r, g, b) {}

    CUDA_CALLABLE Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b);
    }
    CUDA_CALLABLE Color& operator+=(const Color& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    CUDA_CALLABLE Color operator/(float t) const {
        float inv = 1.0f / t;
        return Color(r * inv, g * inv, b * inv);
    }
     void to_uint8(uint8_t& red, uint8_t& green, uint8_t& blue) const {
        red = static_cast<uint8_t>(255.999f * r);
        green = static_cast<uint8_t>(255.999f * g);
        blue = static_cast<uint8_t>(255.999f * b);
    }
};



 CUDA_CALLABLE inline Color operator*(const Color& c, float t) {
    return Color(c.r * t, c.g * t, c.b * t);
}
CUDA_CALLABLE inline Color operator*(float t, const Color& c) {
    return c * t;
}
/*CUDA_CALLABLE inline vecteur3d operator+(const vecteur3d& a, const vecteur3d& b) {
    return vecteur3d(a.x + b.x, a.y + b.y, a.z + b.z);
}*/
CUDA_CALLABLE inline Color operator+(const Color& a, const Color& b) {
    return Color(a.r + b.r, a.g + b.g, a.b + b.b);
}
#undef CUDA_CALLABLE 
#endif