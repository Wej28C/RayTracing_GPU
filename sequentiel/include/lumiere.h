#ifndef LUMIERE_H
#define LUMIERE_H

#include "point3d.h"
#include "color.h"

class Lumiere {
public:
    point3d position;
    Color color;

    Lumiere(const point3d& pos = point3d(0,0,0), const Color& col = Color(1, 1, 1))
        : position(pos), color(col) {}
};

#endif
