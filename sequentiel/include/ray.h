#ifndef RAY_H
#define RAY_H

#include "vecteur3d.h"
#include "point3d.h"

class Ray {
  private:
    point3d orig;
    vecteur3d dir;
  public:
    Ray() {}

    Ray(const point3d& origin, const vecteur3d& direction) : orig(origin), dir(direction) {}

    const point3d& origin() const  { return orig; }
    const vecteur3d& direction() const { return dir; }

    point3d emis(float t) const {
        return orig + t*dir;
    }
};

#endif