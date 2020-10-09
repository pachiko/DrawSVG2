#include "texture.h"
#include "color.h"

#include <assert.h>
#include <iostream>
#include <algorithm>

using namespace std;

namespace CMU462 {

inline void uint8_to_float( float dst[4], unsigned char* src ) {
  uint8_t* src_uint8 = (uint8_t *)src;
  dst[0] = src_uint8[0] / 255.f;
  dst[1] = src_uint8[1] / 255.f;
  dst[2] = src_uint8[2] / 255.f;
  dst[3] = src_uint8[3] / 255.f;
}

inline void float_to_uint8( unsigned char* dst, float src[4] ) {
  uint8_t* dst_uint8 = (uint8_t *)dst;
  dst_uint8[0] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[0])));
  dst_uint8[1] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[1])));
  dst_uint8[2] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[2])));
  dst_uint8[3] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[3])));
}

void Sampler2DImp::generate_mips(Texture& tex, int startLevel) {

  // NOTE: 
  // This starter code allocates the mip levels and generates a level 
  // map by filling each level with placeholder data in the form of a 
  // color that differs from its neighbours'. You should instead fill
  // with the correct data!

  // Task 7: Implement this

  // check start level
  if ( startLevel >= tex.mipmap.size() ) {
    std::cerr << "Invalid start level"; 
  }

  // allocate sublevels
  int baseWidth  = tex.mipmap[startLevel].width;
  int baseHeight = tex.mipmap[startLevel].height;
  // L = sqrt(max(Lx**2, Ly**2))
  int numSubLevels = (int)(log2f( (float)max(baseWidth, baseHeight)));
  // d= log2(L)
  numSubLevels = min(numSubLevels, kMaxMipLevels - startLevel - 1);
  tex.mipmap.resize(startLevel + numSubLevels + 1);

  int width  = baseWidth;
  int height = baseHeight;
  for (int i = 1; i <= numSubLevels; i++) {

    MipLevel& level = tex.mipmap[startLevel + i];

    // handle odd size texture by rounding down
    width  = max( 1, width  / 2); assert(width  > 0);
    height = max( 1, height / 2); assert(height > 0);

    level.width = width;
    level.height = height;
    level.texels = vector<unsigned char>(4 * width * height);

  }

  for(size_t a = 1; a < tex.mipmap.size(); ++a) { // which level
        MipLevel& mip = tex.mipmap[a];
        MipLevel& larger = tex.mipmap[a - 1];
        
        // same algorithm as resolve()
        for(size_t i = 0; i < larger.height; i+=2) {
            for(size_t j = 0; j < larger.width; j+=2) {
                int accum[] = { 0, 0, 0, 0 };

                for (size_t k = 0; k < 4; k++) { // 4 neighbours
                    for (size_t l = 0; l < 4; l++) { // RGBA
                        accum[l] += larger.texels[4* ((i + k/2) * larger.width + j + k%2) + l];
                    }
                }

                for (size_t l = 0; l < 4; l++) { // RGBA
                    accum[l] /= 4;
                    mip.texels[4 * (i/2 * mip.width + j/2) + l] = accum[l];
                }
            }
        }
  }
}

Color Sampler2DImp::sample_nearest(Texture& tex, 
                                   float u, float v, 
                                   int level) {

  // Task 6: Implement nearest neighbour interpolation
  
  // return magenta for invalid level
    if (level >= tex.mipmap.size())
        return Color(1,0,1,1);

    Color c = Color();
    MipLevel& mip = tex.mipmap[level];
    vector<unsigned char>& texels = mip.texels;

    int x = (int) max(0.0f, floor(u * mip.width - 0.5f));
    int y = (int) max(0.0f, floor(v * mip.height - 0.5f));

    int index = 4 * (x + y * mip.width);
    uint8_to_float(&c.r, &texels[index]);
    return c;
}

Color Sampler2DImp::sample_bilinear(Texture& tex, 
                                    float u, float v, 
                                    int level) {
  
  // Task 6: Implement bilinear filtering

  // return magenta for invalid level
    if (level >= tex.mipmap.size())
        return Color(1, 0, 1, 1);

    Color c = Color();
    MipLevel& mip = tex.mipmap[level];
    vector<unsigned char>& texels = mip.texels;

    // sample point
    float x = u * mip.width;
    float y = v * mip.height;

    // lower coords: 1.8 -> 1, 0.2 -> 0
    int lx =  max(0, (int) floor(x - 0.5f));
    int ly =  max(0, (int) floor(y - 0.5f));

    // upper coords: lower coords + 1, but not out-of-bounds
    int ux = min((int) mip.width - 1, lx + 1);
    int uy = min((int) mip.height - 1, ly + 1);

    // weights: 1.8 - 1.5 -> 0.3; 3.2 - 2.5 -> 0.7
    float s = clamp(x - lx - 0.5f, 0.0f, 1.0f);
    float t = clamp(y - ly - 0.5f, 0.0f, 1.0f);

    Color c1 = Color();
    Color c2 = Color();
    Color c3 = Color();
    Color c4 = Color();
    uint8_to_float(&c1.r, &texels[4 * (lx + ly * mip.width)]);
    uint8_to_float(&c2.r, &texels[4 * (ux + ly * mip.width)]);
    uint8_to_float(&c3.r, &texels[4 * (lx + uy * mip.width)]);
    uint8_to_float(&c4.r, &texels[4 * (ux + uy * mip.width)]);

    // BiLerp
    c = (c1 * (1.0f - s) + c2 * s) * (1 - t) + (c3 * (1.0f - s) + c4 * s) * t;
    return c;
}

Color Sampler2DImp::sample_trilinear(Texture& tex, 
                                     float u, float v, 
                                     float u_scale, float v_scale) {

  // Task 7: Implement trilinear filtering

    float dudx = tex.width / u_scale;
    float dvdx = tex.height / u_scale;
    float dudy = tex.width / v_scale;
    float dvdy = tex.height / v_scale;

    float Lx2 = dudx * dudx + dvdx * dvdx;
    float Ly2 = dudy * dudy + dvdy * dvdy;

    float w = max(0.0f, log2f(sqrt(max(Lx2, Ly2)))); // IT IS VERY IMPORTANT TO CHECK BOUNDS
    int d = int(w);
    w = w - d;

    if (d >= tex.mipmap.size())
        return Color(1,0,1,1);
     
    Color a = this->sample_bilinear(tex, u, v, d);
    if ((d + 1) >= tex.mipmap.size())
        return a;

    Color b = this->sample_bilinear(tex, u, v, d + 1);
    return (1.0f - w) * a + (w) * b;
}

} // namespace CMU462
