#ifndef POINT3D_H
#define POINT3D_H

#include "vecteur3d.h"
#include <iostream>
#include <cmath>

class point3d
{
public:
    // c tableau pour les coordonnées x,y,z
    float c[3];
    
    // Constructeurs
    point3d(): c{0, 0, 0} {};
    point3d(float x, float y, float z): c{x, y, z} {};
    
    // Accesseurs
    float getX() const { return c[0]; }
    float getY() const { return c[1]; }
    float getZ() const { return c[2]; }
    
    // Opérateurs
    point3d operator-() const { return point3d(-c[0], -c[1], -c[2]); }
    double operator[](int i) const { return c[i]; }
    
    // Opérations avec d'autres points
    point3d& operator+=(const vecteur3d& v) {
        c[0] += v.c[0];
        c[1] += v.c[1];
        c[2] += v.c[2];
        return *this;
    }
    
    point3d& operator-=(const vecteur3d& v) {
        c[0] -= v.c[0];
        c[1] -= v.c[1];
        c[2] -= v.c[2];
        return *this;
    }
    
    // Opérations entre points (retournent des vecteurs)
    vecteur3d operator-(const point3d& p) const {
        return vecteur3d(c[0] - p.c[0], c[1] - p.c[1], c[2] - p.c[2]);
    }
    
    point3d operator+(const vecteur3d& v) const {
        return point3d(c[0] + v.c[0], c[1] + v.c[1], c[2] + v.c[2]);
    }
    
    ~point3d() {};
    
    // Distance entre deux points
    float distance(const point3d& p) const {
        float dx = c[0] - p.c[0];
        float dy = c[1] - p.c[1];
        float dz = c[2] - p.c[2];
        return sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    // Affichage
    void print() const {
        std::cout << "(" << c[0] << ", " << c[1] << ", " << c[2] << ")\n";
    }
};



#endif // POINT3D_H