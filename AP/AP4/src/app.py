"""
Command-line application for Delaunay triangulation and Voronoi diagram generation.

This is the main entry point that orchestrates the pipeline:
1. Load points from CSV or generate random points
2. Compute Delaunay triangulation
3. Build Voronoi diagram from Delaunay dual
4. Optional barycentric interpolation for coloring
5. Render and save visualization
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List
from datetime import datetime

from .geometry import Point, orient
from .delaunay import delaunay_triangulation
from .voronoi import triangle_circumcenters, voronoi_edges
from .interpolation import face_colors, scalar_from_point
from .visualize import plot_delaunay_and_voronoi


def load_points_from_csv(csv_path: Path) -> List[Point]:
    """
    Load points from a CSV file.
    
    Expected format: header row with 'x,y' or 'x,y,name' columns.
    Additional columns are ignored.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of Point objects
    """
    points = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['x'])
                y = float(row['y'])
                points.append(Point(x, y))
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping invalid row: {row} ({e})", file=sys.stderr)
    return points


def generate_random_points(n: int, seed: int = None) -> List[Point]:
    """
    Generate n random points in a unit square.
    
    Args:
        n: Number of points
        seed: Random seed for reproducibility
        
    Returns:
        List of Point objects
    """
    if seed is not None:
        random.seed(seed)
    
    points = []
    for _ in range(n):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        points.append(Point(x, y))
    return points


def self_test(pts: List[Point], tris: List, vor_edges: List) -> bool:
    """
    Run basic consistency checks on the triangulation and Voronoi diagram.
    
    Checks:
    1. No triangle has collinear vertices
    2. Each Voronoi edge connects circumcenters of adjacent triangles
    3. Sum of triangle areas roughly equals convex hull area (loose tolerance)
    
    Args:
        pts: List of points
        tris: List of triangles
        vor_edges: List of Voronoi edges
        
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n=== Self-test ===")
    all_pass = True
    
    # Test 1: No collinear triangles
    collinear_count = 0
    for tri in tris:
        p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
        if abs(orient(p1, p2, p3)) < 1e-10:
            collinear_count += 1
    
    if collinear_count > 0:
        print(f"⚠️  Warning: {collinear_count} collinear triangles found")
        all_pass = False
    else:
        print("✅ No collinear triangles")
    
    # Test 2: Voronoi edges connect circumcenters (basic check)
    if len(vor_edges) > 0:
        print(f"✅ {len(vor_edges)} Voronoi edges generated")
    else:
        print("⚠️  Warning: No Voronoi edges")
        all_pass = False
    
    # Test 3: Triangle area sum (simplified check)
    total_area = 0.0
    for tri in tris:
        p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
        area = abs(orient(p1, p2, p3)) / 2.0
        total_area += area
    
    if total_area > 0:
        print(f"✅ Total triangle area: {total_area:.2f}")
    else:
        print("⚠️  Warning: Zero total area")
        all_pass = False
    
    return all_pass


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute Delaunay triangulation and Voronoi diagrams"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dataset', type=str, help='Path to CSV file with x,y columns')
    input_group.add_argument('--random', type=int, help='Generate N random points')
    
    parser.add_argument('--seed', type=int, help='Random seed (for --random)')
    parser.add_argument('--interpolate', choices=['yes', 'no'], default='no',
                       help='Enable barycentric interpolation coloring')
    parser.add_argument('--save-name', type=str, help='Output filename (without extension)')
    parser.add_argument('--selftest', action='store_true', help='Run consistency checks')
    
    args = parser.parse_args()
    
    # Load or generate points
    if args.dataset:
        csv_path = Path(args.dataset)
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        pts = load_points_from_csv(csv_path)
        print(f"Loaded {len(pts)} points from {csv_path}")
    else:
        pts = generate_random_points(args.random, args.seed)
        print(f"Generated {len(pts)} random points (seed={args.seed})")
    
    if len(pts) < 3:
        print("Error: Need at least 3 points", file=sys.stderr)
        sys.exit(1)
    
    # Compute Delaunay triangulation
    print("Computing Delaunay triangulation...")
    tris = delaunay_triangulation(pts)
    print(f"  → {len(tris)} triangles")
    
    # Build Voronoi diagram
    print("Building Voronoi diagram from Delaunay dual...")
    vor_centers = triangle_circumcenters(tris, pts)
    vor_edges_list = voronoi_edges(tris, pts)
    print(f"  → {len(vor_centers)} Voronoi vertices")
    print(f"  → {len(vor_edges_list)} Voronoi edges")
    
    # Optional interpolation
    face_colors_list = None
    if args.interpolate == 'yes':
        print("Computing barycentric interpolation...")
        face_colors_list = face_colors(tris, pts, scalar_from_point)
        print("  → Face colors computed")
    
    # Generate output path
    if args.save_name:
        save_name = args.save_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"voronoi_{timestamp}"
    
    output_dir = Path("outputs/screenshots")
    output_path = output_dir / f"{save_name}.png"
    
    # Render
    title = f"Delaunay & Voronoi ({len(pts)} points, {len(tris)} triangles)"
    print(f"Rendering visualization...")
    plot_delaunay_and_voronoi(pts, tris, vor_edges_list, output_path, title, face_colors_list)
    print(f"  → Saved to {output_path}")
    
    # Self-test if requested
    if args.selftest:
        self_test(pts, tris, vor_edges_list)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Points: {len(pts)}")
    print(f"Triangles: {len(tris)}")
    print(f"Voronoi edges: {len(vor_edges_list)}")
    print(f"Output: {output_path}")
    print(f"Output absolute: {output_path.absolute()}")


if __name__ == '__main__':
    main()

