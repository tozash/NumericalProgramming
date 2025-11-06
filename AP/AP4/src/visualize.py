"""
Visualization of Delaunay triangulation and Voronoi diagrams.

Supports matplotlib (preferred) with automatic fallback to pure-Python SVG export
if matplotlib is unavailable. This ensures the project runs with only the standard library.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from .geometry import Point, Triangle


def plot_delaunay_and_voronoi(pts: List[Point], tris: List[Triangle], 
                              vor_edges: List[Tuple[Point, Point]],
                              save_path: Path, title: str = "Delaunay & Voronoi",
                              face_colors: Optional[List[float]] = None) -> Path:
    """
    Plot Delaunay triangulation and Voronoi diagram on the same figure.
    
    Args:
        pts: List of points (sites)
        tris: List of Delaunay triangles
        vor_edges: List of Voronoi edges as (Point, Point) tuples
        save_path: Path to save the figure
        title: Figure title
        face_colors: Optional list of scalar values for triangle coloring
        
    Returns:
        Path to saved figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.collections import LineCollection
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
    
    if HAS_MATPLOTLIB:
        return _plot_matplotlib(pts, tris, vor_edges, save_path, title, face_colors)
    else:
        return _plot_svg(pts, tris, vor_edges, save_path, title, face_colors)


def _plot_matplotlib(pts: List[Point], tris: List[Triangle],
                    vor_edges: List[Tuple[Point, Point]],
                    save_path: Path, title: str,
                    face_colors: Optional[List[float]] = None) -> Path:
    """Matplotlib implementation."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import LineCollection
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Delaunay triangles
    if face_colors is not None:
        # Color-coded triangles
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=min(face_colors), vmax=max(face_colors))
        cmap = cm.viridis
        
        for tri, color_val in zip(tris, face_colors):
            p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
            triangle = patches.Polygon(
                [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)],
                closed=True, alpha=0.3, edgecolor='gray', linewidth=0.5,
                facecolor=cmap(norm(color_val))
            )
            ax.add_patch(triangle)
    else:
        # Simple gray triangles
        for tri in tris:
            p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
            triangle = patches.Polygon(
                [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)],
                closed=True, fill=False, edgecolor='lightgray', linewidth=0.5
            )
            ax.add_patch(triangle)
    
    # Plot Voronoi edges
    vor_lines = [[(e[0].x, e[0].y), (e[1].x, e[1].y)] for e in vor_edges]
    lc = LineCollection(vor_lines, colors='black', linewidths=1.5, alpha=0.7)
    ax.add_collection(lc)
    
    # Plot points
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    ax.scatter(xs, ys, c='red', s=50, zorder=5, edgecolors='darkred', linewidths=1)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits - zoom in on data points with small padding
    if pts:
        # Focus on the actual data points, not the extended Voronoi edges
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        
        # Add padding (50% of range) to show more context
        dx = maxx - minx
        dy = maxy - miny
        padding_x = max(dx * 0.50, 0.05) if dx > 0 else 0.1
        padding_y = max(dy * 0.50, 0.05) if dy > 0 else 0.1
        
        ax.set_xlim(minx - padding_x, maxx + padding_x)
        ax.set_ylim(miny - padding_y, maxy + padding_y)
    
    plt.tight_layout()
    # Re-apply limits after tight_layout to ensure they're respected
    if pts:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = maxx - minx
        dy = maxy - miny
        padding_x = max(dx * 0.50, 0.05) if dx > 0 else 0.1
        padding_y = max(dy * 0.50, 0.05) if dy > 0 else 0.1
        ax.set_xlim(minx - padding_x, maxx + padding_x)
        ax.set_ylim(miny - padding_y, maxy + padding_y)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def _plot_svg(pts: List[Point], tris: List[Triangle],
              vor_edges: List[Tuple[Point, Point]],
              save_path: Path, title: str,
              face_colors: Optional[List[float]] = None) -> Path:
    """Pure-Python SVG fallback (no external dependencies)."""
    if not pts:
        return save_path
    
    # Compute bounding box
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    minx, miny = min(xs), min(ys)
    maxx, maxy = max(xs), max(ys)
    
    # Add padding
    dx, dy = maxx - minx, maxy - miny
    padding = max(dx, dy) * 0.1
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    
    width = 800
    height = 800
    scale_x = width / (maxx - minx) if maxx > minx else 1
    scale_y = height / (maxy - miny) if maxy > miny else 1
    scale = min(scale_x, scale_y)
    
    def to_svg_x(x):
        return (x - minx) * scale
    
    def to_svg_y(y):
        return height - (y - miny) * scale  # Flip Y for SVG coordinates
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
        f.write(f'<title>{title}</title>\n')
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        
        # Draw Delaunay triangles
        f.write('<g id="delaunay" stroke="lightgray" stroke-width="1" fill="none">\n')
        for tri in tris:
            p1, p2, p3 = pts[tri.i], pts[tri.j], pts[tri.k]
            f.write(f'<polygon points="{to_svg_x(p1.x)},{to_svg_y(p1.y)} '
                   f'{to_svg_x(p2.x)},{to_svg_y(p2.y)} '
                   f'{to_svg_x(p3.x)},{to_svg_y(p3.y)}"/>\n')
        f.write('</g>\n')
        
        # Draw Voronoi edges
        f.write('<g id="voronoi" stroke="black" stroke-width="2" fill="none">\n')
        for e in vor_edges:
            f.write(f'<line x1="{to_svg_x(e[0].x)}" y1="{to_svg_y(e[0].y)}" '
                   f'x2="{to_svg_x(e[1].x)}" y2="{to_svg_y(e[1].y)}"/>\n')
        f.write('</g>\n')
        
        # Draw points
        f.write('<g id="points" fill="red" stroke="darkred" stroke-width="1">\n')
        for p in pts:
            f.write(f'<circle cx="{to_svg_x(p.x)}" cy="{to_svg_y(p.y)}" r="3"/>\n')
        f.write('</g>\n')
        
        f.write('</svg>\n')
    
    return save_path

