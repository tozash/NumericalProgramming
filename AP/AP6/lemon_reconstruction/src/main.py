import argparse
import os
import sys

from image_io import load_image, preprocess_image
from edges import compute_gradients, compute_magnitude, get_edge_map
from axis import find_symmetry_axis
from profile import extract_profile
from fit_curve import fit_spline, fit_bezier
from volume import integrate_volume
from visualize import plot_edges, plot_axis, plot_profile, plot_3d_surface
from utils import log

def main():
    parser = argparse.ArgumentParser(description="Lemon Reconstruction from single image.")
    parser.add_argument("--image", default=os.path.join("assets", "input_lemon.jpg"), help="Path to input image")
    parser.add_argument("--scale", type=float, default=None, help="Scale in cm per pixel. If not given, calculates in pixel units.")
    parser.add_argument("--method", choices=["spline", "bezier"], default="spline", help="Approximation method")
    
    args = parser.parse_args()
    
    # Ensure output dir
    output_dir = os.path.join("assets", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    log("Step 1", f"Loading image from {args.image}")
    try:
        raw_img = load_image(args.image)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Preprocess
    log("Step 2", "Preprocessing (Grayscale + Gaussian Blur)...")
    gray_img = preprocess_image(raw_img)
    
    # Edge Detection
    log("Step 3", "Computing Gradients and Edges...")
    Ix, Iy = compute_gradients(gray_img)
    magnitude = compute_magnitude(Ix, Iy)
    edge_map = get_edge_map(magnitude)
    plot_edges(edge_map, os.path.join(output_dir, "edges.png"))
    
    # Axis Detection
    log("Step 4", "Detecting Symmetry Axis...")
    axis_x = find_symmetry_axis(edge_map)
    log("Step 4", f"Found axis at x = {axis_x}")
    plot_axis(raw_img, axis_x, os.path.join(output_dir, "axis_overlay.png"))
    
    # Profile Extraction
    log("Step 5", "Extracting Profile...")
    y_raw, r_raw = extract_profile(edge_map, axis_x)
    log("Step 5", f"Extracted {len(y_raw)} profile points.")
    
    # Curve Fitting
    log("Step 6", f"Fitting Curve using {args.method} method...")
    if args.method == "spline":
        r_fit, fit_func = fit_spline(y_raw, r_raw)
    else:
        r_fit, fit_func = fit_bezier(y_raw, r_raw)
        
    plot_profile(y_raw, r_raw, y_raw, r_fit, os.path.join(output_dir, "fitted_profile.png"))
    
    # 3D Reconstruction Visualization
    log("Step 7", "Generating 3D Surface...")
    plot_3d_surface(y_raw, r_fit, os.path.join(output_dir, "surface3d.png"))
    
    # Volume Calculation
    log("Step 8", "Computing Volume...")
    vol_pixels = integrate_volume(y_raw, r_fit)
    
    print("-" * 30)
    print(f"Volume (pixels^3): {vol_pixels:.2f}")
    if args.scale:
        # Volume in cm^3 = vol_pixels * (scale_cm_per_pixel)^3
        vol_cm3 = vol_pixels * (args.scale ** 3)
        print(f"Scale: {args.scale} cm/pixel")
        print(f"Volume (cm^3): {vol_cm3:.4f}")
    else:
        print("Volume in relative pixel units (provide --scale to convert).")
    print("-" * 30)
    
    log("Final", "Done. Check assets/outputs/ for results.")

if __name__ == "__main__":
    main()
