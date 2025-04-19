#ifndef OBJET_H
#define OBJET_H

//#include "ray.h"
//#include "color.h"
#include "InfoIntersect.h"
class Material;
class Objet {
public:
    
    //retourne vrai si le rayon touche l’objet (et met à jour la distance t
    virtual bool intersect(const Ray& ray, float& t_min, float& t_max, InfoIntersect& inf) const = 0;
    virtual ~Objet() = default ;
  
};

#endif 
