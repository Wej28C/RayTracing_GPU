#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Objet.h"
#include "point3d.h"
#include "vecteur3d.h"
class Triangle : public Objet {
    point3d v0, v1, v2;
    vecteur3d normal;
    std::shared_ptr<Material> material;

public:
    Triangle(const point3d& a, const point3d& b, const point3d& c, std::shared_ptr<Material> m)
        : v0(a), v1(b), v2(c), material(m) {
        vecteur3d edge1 = v1 - v0;
        vecteur3d edge2 = v2 - v0;
        vecteur3d temp = temp.cross(edge1, edge2);
        normal = unit_vector(temp);
    }

    bool intersect(const Ray& ray, float& t_min, float& t_max, InfoIntersect& rec) const override {
        vecteur3d edge1 = v1 - v0;
        vecteur3d edge2 = v2 - v0;
        vecteur3d h = h.cross(ray.direction(), edge2);
        float a = dot(edge1, h);

        if (a > -1e-6 && a < 1e-6) return false;

        float f = 1.0f / a;
        vecteur3d s = ray.origin() - v0;
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        vecteur3d q = q.cross(s, edge1);
        float v = f * dot(ray.direction(), q);

        if (v < 0.0f || u + v > 1.0f) return false;

        float t = f * dot(edge2, q);
        if (t < t_min || t > t_max) return false;

        rec.t = t;
        rec.p = ray.emis(t);
        rec.material = material;
        rec.set_face_normal(ray, normal);
        return true;
    }
};

#endif // TRIANGLE_H