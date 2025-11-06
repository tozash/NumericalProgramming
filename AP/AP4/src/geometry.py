"""
Geometric primitives and helper functions for Delaunay triangulation and Voronoi diagrams.

This module provides the fundamental data structures (Point, Edge, Triangle) and
pure-math utilities needed for computational geometry operations.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Point:
    """A 2D point with x and y coordinates."""
    x: float
    y: float


@dataclass
class Edge:
    """
    An edge connecting two points by their indices.
    Stored in canonical order (min index, max index) for easy comparison.
    """
    a: int  # index of first point (min)
    b: int  # index of second point (max)

    def __post_init__(self):
        """Ensure canonical ordering: a < b."""
        if self.a > self.b:
            self.a, self.b = self.b, self.a

    def __hash__(self):
        return hash((self.a, self.b))

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b


@dataclass
class Triangle:
    """
    A triangle defined by three point indices (i, j, k).
    Counter-clockwise (CCW) orientation is preferred for consistent computations.
    """
    i: int
    j: int
    k: int

    def edges(self) -> List[Edge]:
        """Return the three edges of this triangle."""
        return [Edge(self.i, self.j), Edge(self.j, self.k), Edge(self.k, self.i)]

    def vertices(self) -> Tuple[int, int, int]:
        """Return the three vertex indices."""
        return (self.i, self.j, self.k)


def orient(a: Point, b: Point, c: Point) -> float:
    """
    Compute the signed area (orientation) of triangle abc.
    
    Returns:
        Positive if abc is counter-clockwise (CCW),
        negative if clockwise (CW),
        zero if collinear.
        
    This is the cross product (b-a) × (c-a) = (bx-ax)(cy-ay) - (by-ay)(cx-ax).
    Used for orientation tests and empty-circumcircle checks.
    """
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def circumcircle(tri: Triangle, pts: List[Point]) -> Tuple[float, float, float]:
    """
    Compute the circumcircle of a triangle.
    
    The circumcircle is the unique circle passing through all three vertices.
    In Delaunay triangulation, the empty-circumcircle property ensures no other
    points lie inside this circle.
    
    Args:
        tri: Triangle with vertex indices
        pts: List of all points
        
    Returns:
        (cx, cy, r2) where (cx, cy) is the center and r2 is the squared radius.
    """
    p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
    
    # Compute circumcenter using perpendicular bisectors
    # Solve: (x-cx)^2 + (y-cy)^2 = r^2 for all three points
    # Using determinant method for stability
    
    d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    
    if abs(d) < 1e-10:
        # Degenerate (collinear) triangle - return a large circle
        cx = (p1.x + p2.x + p3.x) / 3
        cy = (p1.y + p2.y + p3.y) / 3
        r2 = 1e10
        return (cx, cy, r2)
    
    ux = (p1.x**2 + p1.y**2) * (p2.y - p3.y)
    ux += (p2.x**2 + p2.y**2) * (p3.y - p1.y)
    ux += (p3.x**2 + p3.y**2) * (p1.y - p2.y)
    
    uy = (p1.x**2 + p1.y**2) * (p3.x - p2.x)
    uy += (p2.x**2 + p2.y**2) * (p1.x - p3.x)
    uy += (p3.x**2 + p3.y**2) * (p2.x - p1.x)
    
    cx = ux / d
    cy = uy / d
    
    # Squared radius
    dx = p1.x - cx
    dy = p1.y - cy
    r2 = dx * dx + dy * dy
    
    return (cx, cy, r2)


def in_circumcircle(p: Point, tri: Triangle, pts: List[Point]) -> bool:
    """
    Test if point p lies inside the circumcircle of triangle tri.
    
    This is the core empty-circumcircle test used in Delaunay triangulation.
    A triangle is "bad" (non-Delaunay) if any point lies inside its circumcircle.
    
    Uses the orientation test: p is inside if the determinant of the 4x4 matrix
    (with homogeneous coordinates) is positive.
    
    Args:
        p: Point to test
        tri: Triangle to check against
        pts: List of all points
        
    Returns:
        True if p is inside the circumcircle (making tri "bad"), False otherwise.
    """
    p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
    
    # Determinant test: |x1 y1 x1^2+y1^2 1|
    #                   |x2 y2 x2^2+y2^2 1|
    #                   |x3 y3 x3^2+y3^2 1|
    #                   |xp yp xp^2+yp^2 1|
    # If positive, p is inside (CCW orientation of p relative to circle)
    
    ax, ay = p1.x - p.x, p1.y - p.y
    bx, by = p2.x - p.x, p2.y - p.y
    cx, cy = p3.x - p.x, p3.y - p.y
    
    det = (ax * (by * (cx*cx + cy*cy) - cy * (bx*bx + by*by)) -
           ay * (bx * (cx*cx + cy*cy) - cx * (bx*bx + by*by)) +
           (ax*ax + ay*ay) * (bx * cy - by * cx))
    
    # For CCW triangle, inside if det > 0
    # Check orientation first
    orient_val = orient(p1, p2, p3)
    if orient_val < 0:
        det = -det  # Flip for CW triangles
    
    return det > 1e-10


def barycentric(p: Point, tri: Triangle, pts: List[Point]) -> Tuple[float, float, float]:
    """
    Compute barycentric coordinates of point p relative to triangle tri.
    
    Barycentric coordinates (u, v, w) represent p as a weighted combination:
    p = u*p1 + v*p2 + w*p3, where u + v + w = 1.
    
    Used for interpolation: if vertices have values v1, v2, v3, then
    interpolated value at p is u*v1 + v*v2 + w*v3.
    
    Args:
        p: Point to compute coordinates for
        tri: Triangle with vertex indices
        pts: List of all points
        
    Returns:
        (u, v, w) barycentric coordinates
    """
    p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
    
    # Area method: u = area(p, p2, p3) / area(p1, p2, p3)
    denom = orient(p1, p2, p3)
    if abs(denom) < 1e-10:
        # Degenerate triangle
        return (1.0/3, 1.0/3, 1.0/3)
    
    u = orient(p, p2, p3) / denom
    v = orient(p1, p, p3) / denom
    w = orient(p1, p2, p) / denom
    
    return (u, v, w)


def bbox(pts: List[Point]) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box of a point set.
    
    Returns:
        (minx, miny, maxx, maxy)
    """
    if not pts:
        return (0.0, 0.0, 1.0, 1.0)
    
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

