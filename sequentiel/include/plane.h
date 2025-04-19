#ifndef PLANE_H
#define PLANE_H

#include "Objet.h"
#include "point3d.h"

class Plane : public Objet {
    point3d point;
    vecteur3d normal;
    std::shared_ptr<Material> material;

public:
    Plane(const point3d& p, const vecteur3d& n, std::shared_ptr<Material> m)
        : point(p), normal(unit_vector(n)), material(m) {}

    bool intersect(const Ray& ray, float& t_min, float& t_max, InfoIntersect& rec) const override {
        float denom = dot(normal, ray.direction());
        if (fabs(denom) < 1e-6) return false;

        float t = dot(point - ray.origin(), normal) / denom;
        if (t < t_min || t > t_max) return false;

        rec.t = t;
        rec.p = ray.emis(t);
        rec.material = material;
        rec.set_face_normal(ray, normal);
        return true;
    }
};

#endif // PLANE_H