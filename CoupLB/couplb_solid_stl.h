#ifndef COUPLB_SOLID_STL_H
#define COUPLB_SOLID_STL_H

#include "couplb_lattice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace LAMMPS_NS {
namespace CoupLB {

struct Vec3 {
  double x, y, z;
};

inline Vec3 vadd(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 vsub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 vmul(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
inline double vdot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 vcross(Vec3 a, Vec3 b) {
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}

struct Triangle {
  Vec3 a, b, c;
};

class STLMesh {
public:
  std::vector<Triangle> tris;

  bool load(const std::string& path, double scale, const double translate[3],
            std::string& error)
  {
    tris.clear();
    if (!load_binary(path, error)) {
      tris.clear();
      if (!load_ascii(path, error)) return false;
    }
    for (auto& t : tris) {
      transform(t.a, scale, translate);
      transform(t.b, scale, translate);
      transform(t.c, scale, translate);
    }
    if (tris.empty()) {
      error = "STL contains no triangles";
      return false;
    }
    return true;
  }

  bool contains(Vec3 p) const
  {
    const Vec3 dir = {0.9341723589627157, 0.2698122951732083, 0.2333729524753242};
    std::vector<double> hits;
    hits.reserve(16);
    for (const auto& tri : tris) {
      double t = 0.0;
      if (ray_triangle(p, dir, tri, t) && t > 1e-10) hits.push_back(t);
    }
    if (hits.empty()) return false;
    std::sort(hits.begin(), hits.end());
    int unique_hits = 0;
    double last = -std::numeric_limits<double>::infinity();
    for (double t : hits) {
      if (unique_hits == 0 || std::fabs(t-last) > 1e-8) {
        unique_hits++;
        last = t;
      }
    }
    return (unique_hits % 2) == 1;
  }

  bool segment_intersection(Vec3 p0, Vec3 p1, double& lambda) const
  {
    const Vec3 d = vsub(p1, p0);
    bool hit = false;
    double best = std::numeric_limits<double>::max();
    for (const auto& tri : tris) {
      double t = 0.0;
      if (!ray_triangle(p0, d, tri, t)) continue;
      if (t >= -1e-10 && t <= 1.0 + 1e-10 && t < best) {
        best = t;
        hit = true;
      }
    }
    if (!hit) return false;
    lambda = std::max(0.0, std::min(1.0, best));
    return true;
  }

private:
  static void transform(Vec3& p, double scale, const double translate[3])
  {
    p.x = p.x * scale + translate[0];
    p.y = p.y * scale + translate[1];
    p.z = p.z * scale + translate[2];
  }

  static bool ray_triangle(Vec3 o, Vec3 d, const Triangle& tri, double& t)
  {
    const double eps = 1e-12;
    const Vec3 e1 = vsub(tri.b, tri.a);
    const Vec3 e2 = vsub(tri.c, tri.a);
    const Vec3 h = vcross(d, e2);
    const double det = vdot(e1, h);
    if (std::fabs(det) < eps) return false;
    const double inv_det = 1.0 / det;
    const Vec3 s = vsub(o, tri.a);
    const double u = inv_det * vdot(s, h);
    if (u < -eps || u > 1.0 + eps) return false;
    const Vec3 q = vcross(s, e1);
    const double v = inv_det * vdot(d, q);
    if (v < -eps || u + v > 1.0 + eps) return false;
    t = inv_det * vdot(e2, q);
    return true;
  }

  static uint32_t read_u32_le(const char* b)
  {
    return (uint32_t)(unsigned char)b[0] |
           ((uint32_t)(unsigned char)b[1] << 8) |
           ((uint32_t)(unsigned char)b[2] << 16) |
           ((uint32_t)(unsigned char)b[3] << 24);
  }

  static float read_f32_le(const char* b)
  {
    uint32_t u = read_u32_le(b);
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
  }

  bool load_binary(const std::string& path, std::string& error)
  {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      error = "cannot open STL file '" + path + "'";
      return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 84) {
      error = "STL file is too small";
      return false;
    }
    in.seekg(80, std::ios::beg);
    char nbuf[4];
    in.read(nbuf, 4);
    if (!in) {
      error = "cannot read binary STL triangle count";
      return false;
    }
    const uint32_t ntri = read_u32_le(nbuf);
    const std::streamoff expected = 84 + (std::streamoff)ntri * 50;
    if (expected != size) {
      error = "not a binary STL with matching size";
      return false;
    }
    tris.reserve(ntri);
    for (uint32_t i = 0; i < ntri; i++) {
      char rec[50];
      in.read(rec, 50);
      if (!in) {
        error = "truncated binary STL triangle data";
        return false;
      }
      Triangle t;
      t.a = {read_f32_le(rec+12), read_f32_le(rec+16), read_f32_le(rec+20)};
      t.b = {read_f32_le(rec+24), read_f32_le(rec+28), read_f32_le(rec+32)};
      t.c = {read_f32_le(rec+36), read_f32_le(rec+40), read_f32_le(rec+44)};
      tris.push_back(t);
    }
    return true;
  }

  bool load_ascii(const std::string& path, std::string& error)
  {
    std::ifstream in(path);
    if (!in) {
      error = "cannot open STL file '" + path + "'";
      return false;
    }
    std::string line;
    std::array<Vec3,3> verts;
    int nv = 0;
    while (std::getline(in, line)) {
      std::istringstream iss(line);
      std::string word;
      iss >> word;
      if (word != "vertex") continue;
      Vec3 p;
      if (!(iss >> p.x >> p.y >> p.z)) {
        error = "malformed ASCII STL vertex line";
        return false;
      }
      verts[nv++] = p;
      if (nv == 3) {
        tris.push_back({verts[0], verts[1], verts[2]});
        nv = 0;
      }
    }
    if (nv != 0) {
      error = "ASCII STL ended inside a triangle";
      return false;
    }
    return !tris.empty();
  }
};

template <typename Lattice>
struct STLApplyStats {
  int solid_nodes = 0;
  int wall_links = 0;
  int missed_links = 0;
};

template <typename Lattice>
STLApplyStats<Lattice> apply_stl_solid(Grid<Lattice>& g, const STLMesh& mesh,
    const double domain_lo[3], double dx_phys, bool mesh_inside_is_solid,
    const double wall_vel_lattice[3])
{
  STLApplyStats<Lattice> stats;
  std::vector<unsigned char> solid(g.ntotal, 0);

  auto pos = [&](int i, int j, int k) {
    return Vec3{
      domain_lo[0] + (g.offset[0] + i - 1) * dx_phys,
      domain_lo[1] + (g.offset[1] + j - 1) * dx_phys,
      domain_lo[2] + (g.offset[2] + k - 1) * dx_phys
    };
  };

  for (int k = 0; k < g.gz; k++) {
    for (int j = 0; j < g.gy; j++) {
      for (int i = 0; i < g.gx; i++) {
        const int n = g.idx(i, j, k);
        const bool inside = mesh.contains(pos(i, j, k));
        solid[n] = (inside == mesh_inside_is_solid) ? 1 : 0;
        if (solid[n] && g.type[n] == 0) {
          g.type[n] = 1;
          stats.solid_nodes++;
        }
      }
    }
  }

  for (int k = 1; k <= g.gz - 2; k++) {
    for (int j = 1; j <= g.gy - 2; j++) {
      for (int i = 1; i <= g.gx - 2; i++) {
        const int n = g.idx(i, j, k);
        if (g.type[n] != 0) continue;
        const Vec3 p0 = pos(i, j, k);
        for (int q = 1; q < Lattice::Q; q++) {
          const int si = i - Lattice::e[q][0];
          const int sj = j - Lattice::e[q][1];
          const int sk = k - Lattice::e[q][2];
          if (si < 0 || si >= g.gx || sj < 0 || sj >= g.gy ||
              sk < 0 || sk >= g.gz) continue;
          const int sn = g.idx(si, sj, sk);
          if (!solid[sn]) continue;
          double lambda = 0.5;
          if (!mesh.segment_intersection(p0, pos(si, sj, sk), lambda))
            stats.missed_links++;
          lambda = std::max(1e-6, std::min(1.0, lambda));
          g.wall_link(q, n) = 1;
          g.wall_dist(q, n) = lambda;
          g.wall_ux(q, n) = wall_vel_lattice[0];
          g.wall_uy(q, n) = wall_vel_lattice[1];
          g.wall_uz(q, n) = wall_vel_lattice[2];
          stats.wall_links++;
        }
      }
    }
  }
  return stats;
}

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_SOLID_STL_H
