#ifndef COLOR_H
#define COLOR_H

// Structure indépendante (pas d'héritage)
struct Color {
    float r, g, b;

    __host__ __device__ Color() : r(0), g(0), b(0) {}
    __host__ __device__ Color(float r, float g, float b) : r(r), g(g), b(b) {}

    __host__ __device__ Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b);
    }
    __host__ void to_uint8(uint8_t& red, uint8_t& green, uint8_t& blue) const {
        red = static_cast<uint8_t>(255.999f * r);
        green = static_cast<uint8_t>(255.999f * g);
        blue = static_cast<uint8_t>(255.999f * b);
    }
};

#endif