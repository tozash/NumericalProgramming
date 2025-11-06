"""
Delaunay triangulation using incremental insertion (Bowyer-Watson algorithm).

The Delaunay triangulation maximizes the minimum angle in all triangles, creating
a "good" mesh. It satisfies the empty-circumcircle property: no point lies inside
the circumcircle of any triangle.
"""

import random
from typing import List, Set
from .geometry import Point, Triangle, Edge, in_circumcircle, bbox


def super_triangle(pts: List[Point]) -> Triangle:
    """
    Create a large triangle that contains all points.
    
    Used as the initial triangle in incremental insertion. All points will be
    inserted inside this triangle, and triangles using super-triangle vertices
    are removed at the end.
    
    Args:
        pts: List of points to enclose
        
    Returns:
        Triangle with indices -1, -2, -3 (virtual vertices)
    """
    if not pts:
        # Default super triangle
        return Triangle(-1, -2, -3)
    
    minx, miny, maxx, maxy = bbox(pts)
    
    # Add padding
    dx = maxx - minx
    dy = maxy - miny
    padding = max(dx, dy) * 2
    
    # Create large triangle vertices (virtual, stored as negative indices)
    # These will be stored in a special way or handled separately
    return Triangle(-1, -2, -3)


def get_super_triangle_points(pts: List[Point]) -> List[Point]:
    """
    Get the actual coordinates of super triangle vertices.
    
    Returns:
        List of 3 Points forming a large triangle
    """
    if not pts:
        return [Point(-10, -10), Point(20, -10), Point(-10, 20)]
    
    minx, miny, maxx, maxy = bbox(pts)
    dx = maxx - minx
    dy = maxy - miny
    padding = max(dx, dy) * 2
    
    return [
        Point(minx - padding, miny - padding),
        Point(maxx + padding, miny - padding),
        Point((minx + maxx) / 2, maxy + padding)
    ]


def delaunay_triangulation(pts: List[Point]) -> List[Triangle]:
    """
    Compute Delaunay triangulation using incremental insertion (Bowyer-Watson).
    
    Algorithm:
    1. Start with a super triangle containing all points
    2. Insert points one by one (shuffled for robustness)
    3. For each point:
       a. Find all "bad" triangles whose circumcircle contains the point
       b. Compute the polygonal cavity boundary (edges not shared by two bad triangles)
       c. Remove bad triangles and retriangulate the cavity with the new point
    4. Remove triangles using super-triangle vertices
    
    The empty-circumcircle test ensures the resulting triangulation is Delaunay.
    
    Args:
        pts: List of points to triangulate
        
    Returns:
        List of Delaunay triangles (using indices into pts)
    """
    if len(pts) < 3:
        return []
    
    # Shuffle for robustness (average-case performance)
    indices = list(range(len(pts)))
    random.shuffle(indices)
    
    # Get super triangle points and add them to extended point list
    super_pts = get_super_triangle_points(pts)
    all_pts = pts + super_pts
    super_indices = [len(pts) + i for i in range(3)]
    
    # Start with super triangle
    triangles = [Triangle(super_indices[0], super_indices[1], super_indices[2])]
    
    # Insert each point
    for idx in indices:
        point = pts[idx]
        
        # Find bad triangles (those whose circumcircle contains the new point)
        bad_triangles = []
        for tri in triangles:
            if in_circumcircle(point, tri, all_pts):
                bad_triangles.append(tri)
        
        # Find cavity boundary edges (edges not shared by two bad triangles)
        edge_count = {}
        for tri in bad_triangles:
            for edge in tri.edges():
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # Remove bad triangles
        triangles = [tri for tri in triangles if tri not in bad_triangles]
        
        # Retriangulate cavity: connect new point to each boundary edge
        for edge in boundary_edges:
            # Create new triangle: edge.a, edge.b, idx
            new_tri = Triangle(edge.a, edge.b, idx)
            triangles.append(new_tri)
    
    # Remove triangles that use super-triangle vertices
    triangles = [tri for tri in triangles 
                 if tri.i < len(pts) and tri.j < len(pts) and tri.k < len(pts)]
    
    return triangles

