#ifndef VECTEUR3D_H
#define VECTEUR3D_H

#include <cuda_runtime.h>
#include <math.h>

struct vecteur3d {
    float x, y, z;

    __host__ __device__ vecteur3d() : x(0), y(0), z(0) {}
    __host__ __device__ vecteur3d(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ vecteur3d operator-() const { 
        return vecteur3d(-x, -y, -z); 
    }

    __host__ __device__ vecteur3d& operator+=(const vecteur3d& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ vecteur3d& operator*=(float t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __host__ __device__ vecteur3d& operator/=(float t) {
        float inv_t = 1.0f / t;
        x *= inv_t;
        y *= inv_t;
        z *= inv_t;
        return *this;
    }

    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ float length_squared() const {
        return x*x + y*y + z*z;
    }

    __host__ __device__ vecteur3d normalized() const {
        float len = length();
        if (len > 0) {
            return vecteur3d(x/len, y/len, z/len);
        }
        return *this;
    }

    __host__ __device__ bool is_zero(float eps = 1e-6f) const {
        return fabsf(x) < eps && fabsf(y) < eps && fabsf(z) < eps;
    }

    __host__ __device__ bool operator==(const vecteur3d& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
};

// Fonctions utilitaires
__host__ __device__ inline float dot(const vecteur3d& u, const vecteur3d& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ inline vecteur3d cross(const vecteur3d& u, const vecteur3d& v) {
    return vecteur3d(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

__host__ __device__ inline vecteur3d operator+(const vecteur3d& a, const vecteur3d& b) {
    return vecteur3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline vecteur3d operator-(const vecteur3d& a, const vecteur3d& b) {
    return vecteur3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline vecteur3d operator*(float t, const vecteur3d& v) {
    return vecteur3d(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ inline vecteur3d operator*(const vecteur3d& v, float t) {
    return t * v;
}

__host__ __device__ inline vecteur3d operator/(const vecteur3d& v, float t) {
    float inv_t = 1.0f / t;
    return vecteur3d(v.x * inv_t, v.y * inv_t, v.z * inv_t);
}

#endif // VECTEUR3D_H