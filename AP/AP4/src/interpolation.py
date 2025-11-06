"""
Barycentric interpolation for scalar fields over Delaunay triangles.

Given a scalar value at each vertex, we can interpolate values anywhere in the
triangulation using barycentric coordinates. This allows us to color triangle
faces or predict scalar field values at arbitrary points.
"""

from typing import List, Callable
from .geometry import Point, Triangle, barycentric


def scalar_from_point(p: Point, landmark: Point = None) -> float:
    """
    Compute a scalar value for a point (default function).
    
    This is a plug-in function that can be replaced. The default uses distance
    to a landmark point or a simple intensity function.
    
    Args:
        p: Point to evaluate
        landmark: Optional landmark point (defaults to origin)
        
    Returns:
        Scalar value (float)
    """
    if landmark is None:
        landmark = Point(0.0, 0.0)
    
    dx = p.x - landmark.x
    dy = p.y - landmark.y
    return (dx * dx + dy * dy) ** 0.5


def interpolate_on_triangle(p: Point, tri: Triangle, pts: List[Point], 
                            values: List[float]) -> float:
    """
    Interpolate a scalar value at point p using barycentric coordinates.
    
    If triangle vertices have values v1, v2, v3, and p has barycentric
    coordinates (u, v, w), then the interpolated value is u*v1 + v*v2 + w*v3.
    
    Args:
        p: Point to interpolate at
        tri: Triangle containing p (or nearby)
        pts: List of all points
        values: Scalar values at each point (indexed by point index)
        
    Returns:
        Interpolated scalar value
    """
    u, v, w = barycentric(p, tri, pts)
    val1 = values[tri.i]
    val2 = values[tri.j]
    val3 = values[tri.k]
    return u * val1 + v * val2 + w * val3


def face_colors(tris: List[Triangle], pts: List[Point], 
                value_fn: Callable[[Point], float] = None) -> List[float]:
    """
    Compute a scalar value (color) for each triangle face.
    
    The value is computed at the triangle's centroid using barycentric interpolation
    from the triangle's vertices.
    
    Args:
        tris: List of triangles
        pts: List of all points
        value_fn: Function mapping Point -> float (defaults to scalar_from_point)
        
    Returns:
        List of scalar values, one per triangle
    """
    if value_fn is None:
        value_fn = scalar_from_point
    
    # Compute values at all points
    values = [value_fn(pts[i]) for i in range(len(pts))]
    
    colors = []
    for tri in tris:
        # Compute centroid
        p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
        centroid = Point((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3)
        
        # Interpolate at centroid
        color = interpolate_on_triangle(centroid, tri, pts, values)
        colors.append(color)
    
    return colors

