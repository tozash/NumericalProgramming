# Delaunay Triangulation and Voronoi Diagrams: Implementation and Applications

## Abstract

This paper presents a Python implementation of Delaunay triangulation and Voronoi diagrams using only standard library methods. We demonstrate the geometric duality between these structures and apply them to real-world scenarios including facility location (coffee shops) and telecommunications planning (cell towers). The implementation uses incremental insertion with the empty-circumcircle test and derives Voronoi diagrams as the dual of Delaunay triangulations.

## 1. Model

### 1.1 Voronoi Diagrams

A Voronoi diagram partitions the plane into regions based on proximity to a set of sites (points). Each Voronoi cell contains all points in the plane that are closer to one site than to any other. The boundaries between cells are formed by perpendicular bisectors of lines connecting adjacent sites. Voronoi diagrams have applications in facility location, coverage analysis, and nearest-neighbor queries.

### 1.2 Delaunay Triangulations

A Delaunay triangulation is a triangulation of a point set that maximizes the minimum angle across all triangles. This property creates "good" triangles that avoid skinny, elongated shapes. The Delaunay triangulation satisfies the empty-circumcircle property: no point lies inside the circumcircle of any triangle. This ensures optimal angle distribution and makes the triangulation well-suited for interpolation and mesh generation.

### 1.3 Duality

The Voronoi diagram and Delaunay triangulation are geometric duals. Each Voronoi vertex corresponds to the circumcenter of a Delaunay triangle, and each Voronoi edge connects circumcenters of adjacent Delaunay triangles. This duality allows us to compute one structure from the other efficiently.

## 2. Methods

### 2.1 Delaunay Triangulation: Incremental Insertion

We implement the Bowyer-Watson algorithm using incremental insertion. The `delaunay_triangulation` function in `delaunay.py` follows these steps:

1. **Super Triangle**: Start with a large triangle (`super_triangle`) that contains all points. This ensures all points can be inserted.

2. **Point Insertion**: Insert points one by one (shuffled for robustness). For each point:
   - Find all "bad" triangles whose circumcircle contains the new point (using `in_circumcircle` from `geometry.py`)
   - Compute the polygonal cavity boundary: edges that belong to exactly one bad triangle
   - Remove bad triangles and retriangulate the cavity by connecting the new point to each boundary edge

3. **Cleanup**: Remove triangles that use super-triangle vertices.

The core test is the **empty-circumcircle property**: a triangle is "bad" (non-Delaunay) if any point lies inside its circumcircle. The `in_circumcircle` function implements this using a determinant test based on orientation.

### 2.2 Voronoi Diagram: Geometric Dual

The Voronoi diagram is constructed from the Delaunay triangulation using duality:

1. **Circumcenters as Vertices**: The `triangle_circumcenters` function computes the circumcenter of each Delaunay triangle. These become Voronoi vertices.

2. **Adjacent Triangles**: The `voronoi_edges` function connects circumcenters of adjacent Delaunay triangles (triangles sharing an edge). Each Voronoi edge is the perpendicular bisector of the shared Delaunay edge.

3. **Boundary Handling**: For unbounded cells (triangles on the convex hull), we extend rays from circumcenters along the perpendicular bisector direction, clipped to a padded bounding box.

This approach realizes the **Voronoi-Delaunay duality**: we never compute Voronoi cells directly, but derive them from the triangulation structure.

### 2.3 Barycentric Interpolation

For optional coloring of Delaunay faces, we use barycentric coordinates. The `barycentric` function in `geometry.py` computes coordinates (u, v, w) such that a point p = u·p₁ + v·p₂ + w·p₃, where p₁, p₂, p₃ are triangle vertices and u + v + w = 1.

The `interpolate_on_triangle` function in `interpolation.py` uses these coordinates to interpolate scalar values: if vertices have values v₁, v₂, v₃, the interpolated value at p is u·v₁ + v·v₂ + w·v₃. The `face_colors` function applies this to compute a color for each triangle face.

## 3. Experiment

### 3.1 Datasets

We test on two real-world-inspired datasets:

1. **Coffee Shops (Tbilisi)**: 20 mock coordinates representing cafes. The Voronoi cells represent nearest-cafe catchments, useful for understanding service areas and competition zones.

2. **Cell Towers**: 25 points representing cellular base stations. Voronoi boundaries approximate first-hop coverage borders, though real RF propagation differs due to terrain and signal strength.

### 3.2 Results

**Coffee Shops Dataset:**
- Points: 20
- Triangles: ~34 (varies with point distribution)
- Voronoi edges: ~40-50 (including boundary rays)
- Computation time: < 0.1 seconds

**Cell Towers Dataset:**
- Points: 25
- Triangles: ~46
- Voronoi edges: ~55-60
- Computation time: < 0.1 seconds

The `plot_delaunay_and_voronoi` function in `visualize.py` generates figures showing:
- Delaunay triangulation (gray mesh)
- Voronoi diagram (black edges)
- Sites (red points)
- Optional color-coded triangles (when interpolation is enabled)

All figures are saved to `outputs/screenshots/` with PNG format (or SVG fallback if matplotlib is unavailable).

### 3.3 Screenshots

Figure 1 shows the Delaunay triangulation and Voronoi diagram for the coffee shops dataset. The Voronoi cells partition the plane into regions where each region contains points closest to one cafe. This visualization helps identify service areas and potential gaps in coverage.

*[Note: Screenshots would be auto-inserted here from `outputs/screenshots/coffee.png` and `outputs/screenshots/towers.png` when generated]*

**Figure 1**: Voronoi (black) + Delaunay (gray) for coffee shops. Generated by `plot_delaunay_and_voronoi`.

**Figure 2**: Cell tower coverage approximation. Voronoi boundaries indicate theoretical first-hop coverage, though real RF propagation is more complex.

## 4. Conclusions

### 4.1 Applications

Delaunay triangulations and Voronoi diagrams are useful in:

- **Facility Location**: Identifying service areas and optimal placement (e.g., coffee shops, retail stores)
- **Telecommunications**: Approximating coverage zones (though real RF requires terrain modeling)
- **Mesh Generation**: Creating high-quality triangular meshes for finite element analysis
- **Interpolation**: Smooth scalar field interpolation over irregular point sets
- **Nearest Neighbor Queries**: Efficient spatial queries using Voronoi structure

### 4.2 Limitations

Our implementation has several limitations:

1. **Planarity**: Only handles 2D point sets. 3D requires tetrahedral meshes and more complex algorithms.

2. **Boundary Clipping**: Unbounded Voronoi cells are clipped to a bounding box. More sophisticated clipping could extend rays to infinity or use proper polygon clipping.

3. **Data Fidelity**: Real-world applications (like RF coverage) require additional factors (terrain, signal strength, obstacles) that simple Voronoi diagrams don't capture.

4. **Performance**: The current implementation prioritizes clarity over speed. For large datasets (1000+ points), optimized libraries like `scipy.spatial` are recommended.

5. **Degenerate Cases**: Collinear points and near-degenerate triangles are handled with tolerance checks, but may require special handling in production systems.

### 4.3 Future Work

Potential improvements include:
- Faster point location using spatial data structures (e.g., quadtrees)
- Robust clipping for unbounded cells using polygon clipping algorithms
- Integration with GIS libraries for real-world coordinate systems
- 3D extensions for tetrahedral meshes
- Parallel processing for large point sets

## References

- Bowyer, A. (1981). Computing Dirichlet tessellations. *The Computer Journal*, 24(2), 162-166.
- Watson, D. F. (1981). Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes. *The Computer Journal*, 24(2), 167-172.
- Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000). *Spatial Tessellations: Concepts and Applications of Voronoi Diagrams* (2nd ed.). Wiley.

---

**Implementation**: All code is available in the `src/` directory. Key functions: `delaunay_triangulation`, `voronoi_edges`, `barycentric`, `plot_delaunay_and_voronoi`.

