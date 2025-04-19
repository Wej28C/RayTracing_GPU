#ifndef MATERIAL_H
#define MATERIAL_H
#include "color.h"
//#include "objet.h"
//#include "InfoIntersect.h"
/*class Material {
public:
    Color color;
    float ambient;   // composante ambiante
    float diffuse;   // composante diffuse
    float specular;  // composante spéculaire
    float shininess; // niveau de brillance

    Material(const Color& col = Color(1,1,1), float a = 0.1f, float d = 0.9f, float s = 0.5f, float sh = 32.0f)
        : color(col), ambient(a), diffuse(d), specular(s), shininess(sh) {}
};*/
struct InfoIntersect;
class Ray;
class Color;

class Material {
    public:
      virtual ~Material() = default;
  
      virtual bool scatter(
          const Ray& r_in, const InfoIntersect& rec, Color& attenuation, Ray& scattered
        ) const {
          return false;
      }
  };
  class lambertian : public Material {
    public:
      lambertian(const Color& albedo) : albedo(albedo) {}
  
      bool scatter(const Ray& r_in, const InfoIntersect& rec, Color& attenuation, Ray& scattered)
      const override {
          auto scatter_direction = rec.normal ;
          // Catch degenerate scatter direction
          if (scatter_direction.near_zero())
              scatter_direction = rec.normal;
  
          scattered = Ray(rec.p, scatter_direction);
          attenuation = albedo;
          return true;
      }
  
    private:
      Color albedo;
  };
  
  
#endif
