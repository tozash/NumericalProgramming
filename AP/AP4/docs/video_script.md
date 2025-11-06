# Video Script: Delaunay Triangulation and Voronoi Diagrams

**Duration**: ~3 minutes  
**Target Audience**: Students and developers interested in computational geometry

---

## 0:00–0:20 Intro

"Hi! Today I'll show you how to compute Delaunay triangulations and Voronoi diagrams using only Python's standard library. These structures are geometric duals: we'll build Delaunay first using the empty-circumcircle test, then derive Voronoi from it. I'll demonstrate on two real-world datasets: coffee shop locations and cell tower placements. Let's dive in!"

---

## 0:20–1:20 Methods

"First, let's look at the core algorithm. In `delaunay.py`, the `delaunay_triangulation` function implements incremental insertion—that's the Bowyer-Watson algorithm. We start with a super triangle, then insert points one by one. For each point, we find all 'bad' triangles whose circumcircle contains it—that's the empty-circumcircle test implemented in `in_circumcircle`. We remove those triangles and retriangulate the cavity.

[Show code snippet of `delaunay_triangulation`]

"Next, in `voronoi.py`, we build the Voronoi diagram as the dual. The `triangle_circumcenters` function computes circumcenters of all Delaunay triangles—these become Voronoi vertices. Then `voronoi_edges` connects circumcenters of adjacent triangles. That's the duality: each Voronoi edge is the perpendicular bisector of a Delaunay edge.

[Show code snippet of `voronoi_edges`]

"Optionally, we can color triangle faces using barycentric interpolation. The `barycentric` function computes coordinates, and `interpolate_on_triangle` blends vertex values. This gives us smooth color gradients over the mesh."

---

## 1:20–2:20 Demo

"Now let's run it! First, the coffee shops dataset:

[Run command: `python -m src.app --dataset src/datasets/coffee_shops_tbilisi.csv --interpolate yes --save-name coffee`]

[Show terminal output and generated plot]

"See how the Voronoi cells partition the plane? Each cell contains points closest to one cafe. This helps identify service areas—if you're in a cell, that's your nearest coffee shop. The Delaunay mesh shows how sites connect, and the colored faces show distance-based intensity.

"Next, the cell towers:

[Run command: `python -m src.app --dataset src/datasets/cell_towers_mock.csv --save-name towers`]

[Show plot]

"Here, Voronoi boundaries approximate first-hop coverage. In reality, RF propagation is more complex—terrain, signal strength, obstacles all matter—but this gives a first-order approximation. Notice how towers near the edge have unbounded cells that extend to the boundary.

"Let's also try random points:

[Run: `python -m src.app --random 50 --seed 42 --save-name random50`]

[Show plot]

"This shows the algorithm works on any point set. The triangulation maximizes minimum angles, avoiding skinny triangles."

---

## 2:20–3:00 Conclusions

"So when would you use this? Voronoi diagrams are great for facility location—finding service areas, optimal placement, coverage analysis. Delaunay triangulations are used in mesh generation, interpolation, and finite element methods. The duality means you can compute one from the other efficiently.

"What could be improved? For large datasets, we'd want faster point location—maybe quadtrees or spatial hashing. For production GIS, we'd integrate with coordinate systems and handle edge cases more robustly. And for real RF coverage, we'd add terrain modeling and signal propagation.

"But for learning the fundamentals, this implementation shows the core ideas clearly: empty-circumcircle test, geometric duality, and barycentric interpolation. Check out the code in `src/` and the paper in `docs/paper.md` for details. Thanks for watching!"

---

**Total word count**: ~420 words  
**Estimated speaking time**: ~3 minutes at normal pace

