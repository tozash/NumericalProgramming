# Delaunay Triangulation and Voronoi Diagrams

A clean Python implementation of Delaunay triangulation and Voronoi diagram construction using only standard library methods and matplotlib (with SVG fallback). This project demonstrates the geometric duality between Delaunay triangulations and Voronoi diagrams, implementing the incremental insertion algorithm (Bowyer-Watson) with the empty-circumcircle test.

## Project Overview

This project computes Delaunay triangulations for 2D point sets using incremental insertion, then derives Voronoi diagrams as the geometric dual. The Voronoi diagram partitions the plane into regions where each region contains all points closer to one site than any other. The implementation includes optional barycentric interpolation for coloring triangle faces based on scalar fields.

**Core Methods:**
- **Delaunay Triangulation**: Incremental insertion (Bowyer-Watson) with empty-circumcircle test
- **Voronoi Diagram**: Built as the geometric dual of Delaunay (Voronoi vertices = triangle circumcenters)
- **Barycentric Interpolation**: Optional coloring of Delaunay faces using barycentric coordinates

## Installation

### Prerequisites
- Python 3.7 or higher

### Setup

1. Create a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install matplotlib (optional, for better visualization):
   ```bash
   pip install matplotlib
   ```
   
   **Note**: If matplotlib is not installed, the project will automatically fall back to pure-Python SVG export (no external dependencies required).

## Quickstart

### Using a Dataset

Run on the coffee shops dataset:
```bash
python -m src.app --dataset src/datasets/coffee_shops_tbilisi.csv --interpolate yes --save-name coffee
```

Run on the cell towers dataset:
```bash
python -m src.app --dataset src/datasets/cell_towers_mock.csv --save-name towers
```

### Generating Random Points

Generate 50 random points with a fixed seed:
```bash
python -m src.app --random 50 --seed 42 --save-name random50
```

### Running Self-Tests

Add the `--selftest` flag to run consistency checks:
```bash
python -m src.app --random 30 --seed 123 --selftest --save-name test
```

## Outputs

All generated figures are saved to `outputs/screenshots/` with PNG format (or SVG if matplotlib is unavailable). The filename is either specified with `--save-name` or auto-generated with a timestamp.

Example output paths:
- `outputs/screenshots/coffee.png`
- `outputs/screenshots/random50.png`

## SVG Fallback (No Dependencies)

If matplotlib is not installed, the project automatically uses a pure-Python SVG exporter. The SVG files can be viewed in any web browser or vector graphics software. To force SVG mode, simply don't install matplotlib.

## Project Structure

```
voronoi-project/
├── src/
│   ├── geometry.py          # Point, Edge, Triangle dataclasses and geometric helpers
│   ├── delaunay.py          # Bowyer-Watson incremental insertion
│   ├── voronoi.py           # Voronoi from Delaunay dual
│   ├── interpolation.py     # Barycentric interpolation
│   ├── visualize.py         # Matplotlib + SVG fallback
│   ├── app.py               # CLI entry point
│   └── datasets/
│       ├── coffee_shops_tbilisi.csv
│       └── cell_towers_mock.csv
├── outputs/
│   └── screenshots/         # Generated figures
├── docs/
│   ├── paper.md             # 2-page technical paper
│   └── video_script.md      # 3-minute demo script
└── README.md
```

## Troubleshooting

### Degenerate/Collinear Points
If you encounter issues with collinear points, the algorithm handles them by returning large circumcircles. For best results, avoid having more than 2 points on the same line.

### Large Scales
For very large point sets (1000+ points), the algorithm may be slow. The current implementation prioritizes clarity over micro-optimizations. For production use, consider libraries like `scipy.spatial` or `shapely`.

### Numeric Stability
The implementation uses tolerance checks (1e-10) for floating-point comparisons. If you encounter precision issues, you may need to adjust these tolerances based on your data scale.

### CSV Format
CSV files must have headers with at least `x,y` columns. Additional columns (like `name` or `tower_id`) are ignored. Example:
```csv
x,y,name
1.0,2.0,Point A
3.0,4.0,Point B
```

## License

This project is provided as-is for educational purposes.

## References

- **Delaunay Triangulation**: Empty-circumcircle property ensures optimal angle distribution
- **Voronoi-Delaunay Duality**: Voronoi vertices are circumcenters of Delaunay triangles
- **Bowyer-Watson Algorithm**: Incremental insertion with cavity retriangulation
- **Barycentric Coordinates**: Linear interpolation over triangles

For detailed methodology, see `docs/paper.md`.

