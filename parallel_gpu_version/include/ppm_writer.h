#ifndef PPM_WRITER_H
#define PPM_WRITER_H
#include <string>
#include "color.h"
void write_ppm(const std::string& filename, const Color* pixels, int width, int height);

#endif