"""
Voronoi diagram construction from Delaunay triangulation via geometric duality.

The Voronoi diagram partitions the plane into regions where each region contains
all points closer to one site than any other. The Voronoi diagram is the geometric
dual of the Delaunay triangulation: Voronoi vertices are circumcenters of Delaunay
triangles, and Voronoi edges connect circumcenters of adjacent triangles.
"""

from typing import List, Dict, Tuple, Set
from .geometry import Point, Triangle, Edge, circumcircle, bbox


def triangle_circumcenters(tris: List[Triangle], pts: List[Point]) -> List[Point]:
    """
    Compute circumcenters of all Delaunay triangles.
    
    These circumcenters become the Voronoi vertices in the dual diagram.
    Each triangle's circumcenter is equidistant from its three vertices.
    
    Args:
        tris: List of Delaunay triangles
        pts: List of all points
        
    Returns:
        List of circumcenter Points (Voronoi vertices)
    """
    centers = []
    for tri in tris:
        cx, cy, _ = circumcircle(tri, pts)
        centers.append(Point(cx, cy))
    return centers


def voronoi_edges(tris: List[Triangle], pts: List[Point]) -> List[Tuple[Point, Point]]:
    """
    Build Voronoi edges by connecting circumcenters of adjacent Delaunay triangles.
    
    Two triangles are adjacent if they share an edge. The Voronoi edge connecting
    their circumcenters is the perpendicular bisector of the shared Delaunay edge.
    
    For unbounded cells (triangles on the convex hull), we clip rays to a padded
    bounding box using the edge's perpendicular bisector direction.
    
    Args:
        tris: List of Delaunay triangles
        pts: List of all points
        
    Returns:
        List of (Point, Point) tuples representing Voronoi edges
    """
    edges = []
    
    # Build adjacency map: edge -> list of triangles sharing it
    edge_to_triangles: Dict[Edge, List[Triangle]] = {}
    for tri in tris:
        for edge in tri.edges():
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(tri)
    
    # Build triangle to index map using tuple of vertices as key (hashable)
    tri_to_index: Dict[Tuple[int, int, int], int] = {}
    for i, tri in enumerate(tris):
        # Use sorted tuple for canonical representation
        key = tuple(sorted(tri.vertices()))
        tri_to_index[key] = i
    
    minx, miny, maxx, maxy = bbox(pts)
    padding = max(maxx - minx, maxy - miny) * 0.5
    
    for edge, edge_tris in edge_to_triangles.items():
        if len(edge_tris) == 2:
            # Interior edge: connect two circumcenters
            tri1, tri2 = edge_tris[0], edge_tris[1]
            key1 = tuple(sorted(tri1.vertices()))
            key2 = tuple(sorted(tri2.vertices()))
            idx1, idx2 = tri_to_index[key1], tri_to_index[key2]
            
            cx1, cy1, _ = circumcircle(tri1, pts)
            cx2, cy2, _ = circumcircle(tri2, pts)
            
            edges.append((Point(cx1, cy1), Point(cx2, cy2)))
        elif len(edge_tris) == 1:
            # Boundary edge: extend ray to bounding box
            tri = edge_tris[0]
            cx, cy, _ = circumcircle(tri, pts)
            
            # Get the two points of the edge
            p1, p2 = pts[edge.a], pts[edge.b]
            
            # Perpendicular bisector direction
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            # Perpendicular: rotate 90 degrees
            perp_x = -dy
            perp_y = dx
            length = (perp_x**2 + perp_y**2)**0.5
            if length > 1e-10:
                perp_x /= length
                perp_y /= length
            
            # Extend in the direction away from the triangle's third vertex
            third_idx = [v for v in tri.vertices() if v != edge.a and v != edge.b][0]
            third_pt = pts[third_idx]
            
            # Choose direction pointing away from third point
            to_third_x = third_pt.x - cx
            to_third_y = third_pt.y - cy
            if perp_x * to_third_x + perp_y * to_third_y > 0:
                perp_x = -perp_x
                perp_y = -perp_y
            
            # Clip to bounding box
            scale = padding * 2
            end_x = cx + perp_x * scale
            end_y = cy + perp_y * scale
            
            edges.append((Point(cx, cy), Point(end_x, end_y)))
    
    return edges


def voronoi_cells(tris: List[Triangle], pts: List[Point]) -> Dict[int, List[Point]]:
    """
    Build Voronoi cells as ordered polygons for each site.
    
    Each cell is the region of points closest to one site. The cell boundary
    is formed by connecting circumcenters of triangles that share the site.
    
    This is a best-effort implementation that orders vertices CCW.
    
    Args:
        tris: List of Delaunay triangles
        pts: List of all points (sites)
        
    Returns:
        Dictionary mapping site index to list of Points forming the cell polygon
    """
    cells: Dict[int, List[Point]] = {i: [] for i in range(len(pts))}
    
    # For each site, collect circumcenters of triangles containing it
    site_to_centers: Dict[int, List[Point]] = {i: [] for i in range(len(pts))}
    
    for tri in tris:
        cx, cy, _ = circumcircle(tri, pts)
        center = Point(cx, cy)
        
        for v in tri.vertices():
            if v < len(pts):  # Not a super-triangle vertex
                site_to_centers[v].append(center)
    
    # For each site, order centers by angle (simple CCW sort)
    for site_idx, centers in site_to_centers.items():
        if not centers:
            continue
        
        # Sort by angle from site
        site_pt = pts[site_idx]
        centers_sorted = sorted(centers, 
                               key=lambda c: (c.x - site_pt.x, c.y - site_pt.y))
        
        # Simple angle-based sort
        def angle_from_site(c: Point) -> float:
            import math
            return math.atan2(c.y - site_pt.y, c.x - site_pt.x)
        
        centers_sorted = sorted(centers, key=angle_from_site)
        cells[site_idx] = centers_sorted
    
    return cells

