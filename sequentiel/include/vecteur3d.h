#ifndef VECTEUR3D_H
#define VECTEUR3D_H
#include <iostream>
#include <cmath>

class vecteur3d
{
private:
    /* data */
public:
    // c tableau pour les coordonnées x,y, z
    float c[3];
    //vecteur3d();
    vecteur3d():c{0,0,0}{};
    vecteur3d(float x, float y, float z):c{x,y,z}{};
    float getX() const { return c[0]; }
    float getY() const { return c[1]; }
    float getZ() const { return c[2]; }

    //opposé
    vecteur3d operator-() const { return vecteur3d(-c[0], -c[1], -c[2]); }
   float operator[](int i) const { return c[i]; }
  //  double& operator[](int i) { return c[i]; }

    vecteur3d& operator+=(const vecteur3d& v) {
        c[0] += v.c[0];
        c[1] += v.c[1];
        c[2] += v.c[2];
        return *this;
    }
  
    vecteur3d& operator-=(const vecteur3d& v) {
        c[0] -= v.c[0];
        c[1] -= v.c[1];
        c[2] -= v.c[2];
        return *this;
    }
    vecteur3d& operator*=(float t) {
       c[0] *= t;
       c[1] *= t;
       c[2] *= t;
        return *this;
    }

    vecteur3d& operator/=(float t) {
        return *this *= 1/t;
    }
   
     // Produit scalaire
     float dot(const vecteur3d& u, const vecteur3d& v) const { 
        return  u.c[0] * v.c[0]
        + u.c[1] * v.c[1]
        + u.c[2] * v.c[2];
    }

    //produit vectoriel
    vecteur3d cross(const vecteur3d& u, const vecteur3d& v) const {
        return vecteur3d(
            u.c[1] * v.c[2] - u.c[2] * v.c[1],
            u.c[2] * v.c[0] - u.c[0] * v.c[2],
            u.c[0] * v.c[1] - u.c[1] * v.c[0]
        );
    }

    float length_squared() const {
        return c[0]*c[0] + c[1]*c[1] + c[2]*c[2];
    }
    //norme (longueur) du vecteur
    float norm() const {
        return sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
    }
   
    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (std::fabs(c[0]) < s) && (std::fabs(c[1]) < s) && (std::fabs(c[2]) < s);
    }
    
    void print() const {
        std::cout << "Vecteur3D(" << c[0] << ", " << c[1] << ", " << c[2] << ")" << std::endl;
    }
    ~vecteur3d()=default;
}; 


/*inline std::ostream& operator<<(std::ostream& out, const vecteur3d& v) {
    return out << v.c[0] << ' ' << v.c[1] << ' ' << v.c[2];
}*/

inline vecteur3d operator+(const vecteur3d& u, const vecteur3d& v) {
    return vecteur3d(u.c[0] + v.c[0], u.c[1] + v.c[1], u.c[2] + v.c[2]);
}

inline vecteur3d operator-(const vecteur3d& u, const vecteur3d& v) {
    return vecteur3d(u.c[0] - v.c[0], u.c[1] - v.c[1], u.c[2] - v.c[2]);
}
//
inline vecteur3d operator*(const vecteur3d& u, const vecteur3d& v) {
    return vecteur3d(u.c[0] * v.c[0], u.c[1] * v.c[1], u.c[2] * v.c[2]);
}
//vecteur * scalaire
inline vecteur3d operator*(float t, const vecteur3d& v) {
    return vecteur3d(t*v.c[0], t*v.c[1], t*v.c[2]);
}

inline vecteur3d operator*(const vecteur3d& v, float t) {
    return t * v;
}

inline vecteur3d operator/(const vecteur3d& v, float t) {
    return (1/t) * v;
}
inline float dot(const vecteur3d& u, const vecteur3d& v) {
    return u.c[0]*v.c[0] + u.c[1]*v.c[1] + u.c[2]*v.c[2];
}
//normalisation du vecteur
inline vecteur3d unit_vector(const vecteur3d& v) {
    return v / v.norm();
}

   
#endif