# Lemon Reconstruction

reconstruct an approximate 3D model of a LEMON from a single 2D side-view image by assuming axial symmetry.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script with an input image:

```bash
python src/main.py --image assets/input_lemon.jpg --scale 0.05 --method spline
```

- `--image`: Path to the input image. (Default: `assets/input_lemon.jpg`)
- `--scale`: Scale in cm per pixel. (Optional, default is pixel units)
- `--method`: Parametric approximation method: `spline` (default) or `bezier`.

## Outputs

Results are saved in `assets/outputs/`:
- `edges.png`: Detected edges.
- `axis_overlay.png`: Image with detected symmetry axis.
- `profile_points.png`: Extracted profile points.
- `fitted_profile.png`: Parametric curve fit.
- `surface3d.png`: 3D reconstruction visualization.

## Structure

- `src/`: Source code.
- `report/`: Project report.
- `assets/`: Input images and generated outputs.
- `video/`: Demo video script.
