#ifndef INFOINTERSECT_H
#define INFOINTERSECT_H
#include <memory>
//#include "material.h"
#include "ray.h"
#include "vecteur3d.h"  
#include "point3d.h"

// Forward declarations
//class Ray;
class Material;

struct InfoIntersect {
    point3d p;
    vecteur3d normal;
   std::shared_ptr<Material> material;
    float t;
    bool front_face;
  //  const Material* material;
    void set_face_normal(const Ray& ray, const vecteur3d& outward_normal) {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : outward_normal * -1;
    }
};
#endif // INFOINTERSECT_H