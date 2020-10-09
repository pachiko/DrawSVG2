#include "software_renderer.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include "triangulation.h"

using namespace std;

namespace CMU462 {


// Implements SoftwareRenderer //

void SoftwareRendererImp::draw_svg( SVG& svg ) {
   // Cleans the sample_buffer
   sample_buffer = vector<unsigned char>(w * h * 4, 255);

  // set top level transformation
  transformation = svg_2_screen;

  // draw all elements
  for ( size_t i = 0; i < svg.elements.size(); ++i ) {
    draw_element(svg.elements[i]);
  }

  // draw canvas outline
  Vector2D a = transform(Vector2D(    0    ,     0    )); a.x--; a.y--;
  Vector2D b = transform(Vector2D(svg.width,     0    )); b.x++; b.y--;
  Vector2D c = transform(Vector2D(    0    ,svg.height)); c.x--; c.y++;
  Vector2D d = transform(Vector2D(svg.width,svg.height)); d.x++; d.y++;

  rasterize_line(a.x, a.y, b.x, b.y, Color::Black);
  rasterize_line(a.x, a.y, c.x, c.y, Color::Black);
  rasterize_line(d.x, d.y, b.x, b.y, Color::Black);
  rasterize_line(d.x, d.y, c.x, c.y, Color::Black);

  // resolve and send to render target
  resolve();
}

// Alpha Blending using Pre-Multiplied Alpha
Color SoftwareRendererImp::alpha_blend(int index, Color over) {
    // read current color from buffer
    Color under = Color();
    under.r = sample_buffer[index] / 255.f;
    under.g = sample_buffer[index + 1] / 255.f;
    under.b = sample_buffer[index + 2] / 255.f;
    under.a = sample_buffer[index + 3] / 255.f;

    // premultiply alphas
    under.r *= under.a;
    under.g *= under.a;
    under.b *= under.a;
    over.r *= over.a;
    over.g *= over.a;
    over.b *= over.a;

    // Blend
    Color blend = over + (1 - over.a) * under;

    // Undo premultiply alpha
    blend.r /= blend.a;
    blend.g /= blend.a;
    blend.b /= blend.a;

    return blend;
}

// Fills a sample in sample_buffer
void SoftwareRendererImp::fill_sample(int sx, int sy, const Color& color) {
    // IT IS VERY IMPORTANT TO CHECK BOUNDS
    if (sx < 0 || sx >= w) return;
    if (sy < 0 || sy >= h) return;

    int index = 4 * (sx + sy * w);
    Color blend = alpha_blend(index, color);

    // fill sample - NOT doing alpha blending!
    sample_buffer[index] = (uint8_t)(blend.r * 255);
    sample_buffer[index + 1] = (uint8_t)(blend.g * 255);
    sample_buffer[index + 2] = (uint8_t)(blend.b * 255);
    sample_buffer[index + 3] = (uint8_t)(blend.a * 255);
}

// Fills a pixel in target_render using the neighbourhood of samples from sample_buffer
// Different from CS248, which fills all samples in the pixel (writes to sample_buffer instead)
void SoftwareRendererImp::fill_pixel(int x, int y, const Color& color) {
    render_target[4 * (x + y * target_w)] = (uint8_t)(color.r * 255);
    render_target[4 * (x + y * target_w) + 1] = (uint8_t)(color.g * 255);
    render_target[4 * (x + y * target_w) + 2] = (uint8_t)(color.b * 255);
    render_target[4 * (x + y * target_w) + 3] = (uint8_t)(color.a * 255);
}

void SoftwareRendererImp::set_sample_rate( size_t sample_rate ) {

  // Task 4: 
  // You may want to modify this for supersampling support
  this->sample_rate = sample_rate;
  this->w = sample_rate * target_w;
  this->h = sample_rate * target_h;
}

void SoftwareRendererImp::set_render_target( unsigned char* render_target,
                                             size_t width, size_t height ) {

  // Task 4: 
  // You may want to modify this for supersampling support
  this->render_target = render_target;
  this->target_w = width;
  this->target_h = height;
  set_sample_rate(sample_rate);
}

void SoftwareRendererImp::draw_element( SVGElement* element ) {

  // Task 5 (part 1):
  // Modify this to implement the transformation stack
  transformation = transformation * element->transform;
  switch(element->type) {
    case POINT:
      draw_point(static_cast<Point&>(*element));
      break;
    case LINE:
      draw_line(static_cast<Line&>(*element));
      break;
    case POLYLINE:
      draw_polyline(static_cast<Polyline&>(*element));
      break;
    case RECT:
      draw_rect(static_cast<Rect&>(*element));
      break;
    case POLYGON:
      draw_polygon(static_cast<Polygon&>(*element));
      break;
    case ELLIPSE:
      draw_ellipse(static_cast<Ellipse&>(*element));
      break;
    case IMAGE:
      draw_image(static_cast<Image&>(*element));
      break;
    case GROUP:
        draw_group(static_cast<Group&>(*element));
      break;
    default:
      break;
  }
  transformation = transformation * element->transform.inv();
}


// Primitive Drawing //

void SoftwareRendererImp::draw_point( Point& point ) {

  Vector2D p = transform(point.position);
  rasterize_point( p.x, p.y, point.style.fillColor );

}

void SoftwareRendererImp::draw_line( Line& line ) { 

  Vector2D p0 = transform(line.from);
  Vector2D p1 = transform(line.to);
  rasterize_line( p0.x, p0.y, p1.x, p1.y, line.style.strokeColor );

}

void SoftwareRendererImp::draw_polyline( Polyline& polyline ) {

  Color c = polyline.style.strokeColor;

  if( c.a != 0 ) {
    int nPoints = polyline.points.size();
    for( int i = 0; i < nPoints - 1; i++ ) {
      Vector2D p0 = transform(polyline.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polyline.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_rect( Rect& rect ) {

  Color c;
  
  // draw as two triangles
  float x = rect.position.x;
  float y = rect.position.y;
  float w = rect.dimension.x;
  float h = rect.dimension.y;

  Vector2D p0 = transform(Vector2D(   x   ,   y   ));
  Vector2D p1 = transform(Vector2D( x + w ,   y   ));
  Vector2D p2 = transform(Vector2D(   x   , y + h ));
  Vector2D p3 = transform(Vector2D( x + w , y + h ));
  
  // draw fill
  c = rect.style.fillColor;
  if (c.a != 0 ) {
    rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    rasterize_triangle( p2.x, p2.y, p1.x, p1.y, p3.x, p3.y, c );
  }

  // draw outline
  c = rect.style.strokeColor;
  if( c.a != 0 ) {
    rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    rasterize_line( p1.x, p1.y, p3.x, p3.y, c );
    rasterize_line( p3.x, p3.y, p2.x, p2.y, c );
    rasterize_line( p2.x, p2.y, p0.x, p0.y, c );
  }

}

void SoftwareRendererImp::draw_polygon( Polygon& polygon ) {

  Color c;

  // draw fill
  c = polygon.style.fillColor;
  if( c.a != 0 ) {

    // triangulate
    vector<Vector2D> triangles;
    triangulate( polygon, triangles );

    // draw as triangles
    for (size_t i = 0; i < triangles.size(); i += 3) {
      Vector2D p0 = transform(triangles[i + 0]);
      Vector2D p1 = transform(triangles[i + 1]);
      Vector2D p2 = transform(triangles[i + 2]);
      rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    }
  }

  // draw outline
  c = polygon.style.strokeColor;
  if( c.a != 0 ) {
    int nPoints = polygon.points.size();
    for( int i = 0; i < nPoints; i++ ) {
      Vector2D p0 = transform(polygon.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polygon.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_ellipse( Ellipse& ellipse ) {

  // Extra credit 

}

void SoftwareRendererImp::draw_image( Image& image ) {

  Vector2D p0 = transform(image.position);
  Vector2D p1 = transform(image.position + image.dimension);

  rasterize_image( p0.x, p0.y, p1.x, p1.y, image.tex );
}

void SoftwareRendererImp::draw_group( Group& group ) {
   
  for ( size_t i = 0; i < group.elements.size(); ++i ) {
    draw_element(group.elements[i]);
  }

}

// Rasterization //

// The input arguments in the rasterization functions 
// below are all defined in screen space coordinates

void SoftwareRendererImp::rasterize_point( float x, float y, Color color ) {

  // fill in the nearest pixel
  int sx = (int) floor(x);
  int sy = (int) floor(y);

  // check bounds
  if ( sx < 0 || sx >= target_w ) return;
  if ( sy < 0 || sy >= target_h ) return;

  // this rasterizes all samples in the pixel (which is what CS248's fill_pixel() does)
  for (int i = 0; i < sample_rate; i++) {
     for (int j = 0; j < sample_rate; j++) {
          fill_sample(sx * sample_rate + i, sy * sample_rate + j, color);
     }
  }
}

void SoftwareRendererImp::rasterize_line( float x0, float y0,
                                          float x1, float y1,
                                          Color color) {

  // Task 2: 
  // Implement line rasterization

    if (abs(y1 - y0) < abs(x1 - x0)) {
        if (x0 > x1)
            plot_line_low(x1, y1, x0, y0, color);
        else
            plot_line_low(x0, y0, x1, y1, color);
    }
    else {
        if (y0 > y1)
            plot_line_high(x1, y1, x0, y0, color);
        else
            plot_line_high(x0, y0, x1, y1, color);
    }
}

void SoftwareRendererImp::plot_line_low(float lx, float ly,
    float ux, float uy, Color color) {
    float dx = ux - lx;
    float dy = uy - ly;
    float yi = 1.0f;

    if (dy < 0.0f) { // go downwards: -1 <= m <= 0
        yi = -1; // up or down 1 pixel?
        dy = -dy; // direction?
    }

    float diff = 2 * dy - dx; // 2*eps*dx + 2*dy < dx; init diff = 2*eps*dx, diff >= 2*dy - dx
    float y = ly;

    for (float x = lx; x <= ux; x++) {
        rasterize_point(x, y, color);

        if (diff > 0) {
            y += yi;
            diff += 2 * (dy - dx); // 2*dx*eps <- 2*dx*eps + 2*dy - 2*dx
        }
        else {
            diff += 2 * dy; // 2*eps*dx <- 2*eps*dx + 2*dy 
        }
    }
}

void SoftwareRendererImp::plot_line_high(float lx, float ly,
    float ux, float uy, Color color) {
    float dx = ux - lx;
    float dy = uy - ly;
    float xi = 1.0f; // swapped the axes for steep gradients

    if (dx < 0.0f) { // go downwards: -Inf <= m <= -1
        xi = -1; // left or right 1 pixel?
        dx = -dx; // direction?
    }

    float diff = 2 * dx - dy;
    float x = lx;

    for (float y = ly; y <= uy; y++) {
        rasterize_point(x, y, color);

        if (diff > 0) {
            x += xi;
            diff += 2 * (dx - dy);
        }
        else {
            diff += 2 * dx;
        }
    }
}

void SoftwareRendererImp::rasterize_triangle( float x0, float y0,
                                              float x1, float y1,
                                              float x2, float y2,
                                              Color color ) {
  // Task 3: 
  // Implement triangle rasterization
    Vector2D vA(x1 - x0, y1 - y0);
    Vector2D vB(x2 - x0, y2 - y0);

    if (cross(vA, vB) < 0) { // swap to CCW; dot product is also fine, but its line with point
        float xtemp = x1, ytemp = y1;
        x1 = x2, y1 = y2;
        x2 = xtemp, y2 = ytemp;
    }

    // Bound of Triangle
    // We must clamp and test each sample within the box here.
    // If we use continuous values (ie floats without clamping), there could be overlap 
    // and samples from one triangle replaces the other triangle
    float xmin = floor(min(x0, min(x1, x2)));
    float ymin = floor(min(y0, min(y1, y2)));
    float xmax = floor(max(x0, max(x1, x2)));
    float ymax = floor(max(y0, max(y1, y2)));

    // 3 Edges
    auto lineTest01 = [&](auto x, auto y) {
        return (x1 - x0) * y + (y0 - y1) * x + (y1 - y0) * x0 - (x1 - x0) * y0;
    };
    auto lineTest12 = [&](auto x, auto y) {
        return (x2 - x1) * y + (y1 - y2) * x + (y2 - y1) * x1 - (x2 - x1) * y1;
    };
    auto lineTest20 = [&](auto x, auto y) {
        return (x0 - x2) * y + (y2 - y0) * x + (y0 - y2) * x2 - (x0 - x2) * y2;
    };

    float period = 1.0f / sample_rate;
    float offset = period * 0.5f;

    // Coverage test for sample points
    for (float y = ymin; y <= ymax; y++) {
        for (float x = xmin; x <= xmax; x++) {
            // This loops through all samples in the pixels (both x & y)
            for (int i = 0; i < sample_rate; i++) {
                for (int j = 0; j < sample_rate; j++) {
                    // sample locations (pixel coordinates)
                    float xs = x + j * period + offset;
                    float ys = y + i * period + offset;
                    if (lineTest01(xs, ys) >= 0 && lineTest12(xs, ys) >= 0 && lineTest20(xs, ys) >= 0) {
                        // fill in the nearest sample in sample buffer
                        int sx = (int)floor(xs * sample_rate);
                        int sy = (int)floor(ys * sample_rate);
                        fill_sample(sx, sy, color);
                    }
                }
            }
        }
    }
    
    // This has some thin white lines
    /*
    for (float y = ymin; y <= ymax; y += period) {
        for (float x = xmin; x <= xmax; x += period) { // The order of traversal is slightly different (all x samples before increment y)
            float xs = x + offset;
            float ys = y + offset;
            if (lineTest01(xs, ys) >= 0 && lineTest12(xs, ys) >= 0 && lineTest20(xs, ys) >= 0) {
                fill_sample((int)floor(xs * sample_rate), (int)floor(ys * sample_rate), color);
            }
        }
    }
    */
}

void SoftwareRendererImp::rasterize_image(float x0, float y0,
                                          float x1, float y1,
                                          Texture& tex) {
    // Task 6: 
    // Implement image rasterization
    float dx = x1 - x0;
    float dy = y1 - y0;

    float period = 1.0f / sample_rate;
    float offset = period * 0.5f;
    float imgW = x1 - x0;
    float imgH = y1 - y0;

    for (float y = floor(y0); y <= floor(y1); y++) {
        for (float x = floor(x0); x <= floor(x1); x++) {
            for (int i = 0; i < sample_rate; i++) {
                for (int j = 0; j < sample_rate; j++) {
                    // sample locations (pixel coordinates)
                    float xs = x + j * period + offset;
                    float ys = y + i * period + offset;
                    
                    float u = (xs - x0) / imgW;
                    float v = (ys - y0) / imgH;
                    Color c = sampler->sample_trilinear(tex, u, v, imgW, imgH);

                    int sx = (int)floor(xs * sample_rate);
                    int sy = (int)floor(ys * sample_rate);
                    fill_sample(sx, sy, c);
                }
            }
        }
    }
}

// resolve samples to render target
void SoftwareRendererImp::resolve( void ) {
  // Task 4: 
  // Implement supersampling
  // You may also need to modify other functions marked with "Task 4".

    // loops through samples
    for (int i = 0; i < h; i += sample_rate) {
        for (int j = 0; j < w; j += sample_rate) {
            int r = 0, g = 0, b = 0, a = 0;
            float num_samples = sample_rate * sample_rate;
            // actual pixel coords
            int x = j / sample_rate;
            int y = i / sample_rate;

            for (int k = 0; k < sample_rate; k++) {
                for (int l = 0; l < sample_rate; l++) {
                    int index = 4 * ((j + l) + (i + k) * w);
                    r += (int) sample_buffer[index];
                    g += (int) sample_buffer[index + 1];
                    b += (int) sample_buffer[index + 2];
                    a += (int) sample_buffer[index + 3];
                }
            }

            r /= num_samples;
            g /= num_samples;
            b /= num_samples;
            a /= num_samples;
            Color avg = Color(float(r / 255.0f), float(g / 255.0f), float(b / 255.0f), float(a / 255.0f));
            fill_pixel(x, y, avg);
        }
    }
}
} // namespace CMU462
