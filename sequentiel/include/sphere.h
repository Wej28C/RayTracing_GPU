#ifndef SPHERE_H
#define SPHERE_H

#include "objet.h"
#include "material.h"
#include "InfoIntersect.h"
class Sphere : public Objet {
    private:
        point3d center;
        float radius;
        std::shared_ptr<Material> material;
       
    public:
        //std::shared_ptr<Material>
    Sphere(point3d center, float radius, std::shared_ptr<Material> m)
        : center(center), radius(radius), material(m) {}

     bool intersect(const Ray& r, float& t_min, float& t_max, InfoIntersect& rec) const override {
   
        vecteur3d oc = r.origin() - center;
        float a = r.direction().length_squared();
        float half_b = dot(oc, r.direction());
        float c = oc.length_squared() - radius*radius;

        float discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;
        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.emis(rec.t);
        vecteur3d outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.material = material;

        return true;
    }


};

#endif // SPHERE_H