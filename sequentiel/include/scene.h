#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include "Objet.h"
#include "Lumiere.h"
#include "color.h"
#include "InfoIntersect.h"
#include "ray.h"
using std::vector;
using std::make_shared;
using std::shared_ptr;

class Scene {
public:
    // Liste d'objets et de lumières dans la scène
    std::vector<shared_ptr<Objet>> objets;
    std::vector<Lumiere> lumieres;
    Scene() {}
    Scene(shared_ptr<Objet> object) { ajouterObjet(object); }
    //void clear() { objets.clear(); }
    // Ajouter un objet à la scène
    void ajouterObjet(shared_ptr<Objet> obj) {
        objets.push_back(obj);
    }

    // Ajouter une lumière à la scène
    void ajouterLumiere(const Lumiere& l) {
        lumieres.push_back(l);
    }

    // Détermine si un rayon touche un objet de la scène et met à jour rec
    bool intersectScene(const Ray& ray, float t_min, float t_max, InfoIntersect& rec) const {
        InfoIntersect temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (const auto& obj : objets) {
            if (obj->intersect(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif

