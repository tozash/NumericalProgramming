
import os
import sys
import numpy as np
import csv

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import get_character_data
from param import chord_length_parameterization
from fit_cubic import fit_cubic_spline
from fit_bspline import fit_bspline, eval_bspline
from metrics import compute_metrics
from plotters import setup_plot_style, plot_comparison

def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, 'outputs', 'plots')
    tables_dir = os.path.join(base_dir, 'outputs', 'tables')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    setup_plot_style()
    
    # Load data
    chars = get_character_data()
    
    # Downsampling experiments
    downsamples = [1, 2, 3]  # Keep every 1st, 2nd, 3rd node
    
    # Results storage
    results = []
    
    print("Starting Spline Experiments...")
    
    for char_name, strokes in chars.items():
        print(f"Processing Character {char_name}...")
        
        for s_idx, stroke in enumerate(strokes):
            stroke_name = f"{char_name}_stroke{s_idx+1}"
            
            # Original full density nodes (Reference)
            orig_x_full = np.array([p[0] for p in stroke])
            orig_y_full = np.array([p[1] for p in stroke])
            # Reference nodes for metric calc are the FULL set
            reference_nodes = list(zip(orig_x_full, orig_y_full))
            
            for d_factor in downsamples:
                print(f"  > Downsample factor: {d_factor}")
                
                # Subset nodes
                subset_indices = list(range(0, len(stroke), d_factor))
                # Ensure last point is included if it's not closed loop 
                # (or even if it is, strict subsetting might miss it)
                # But for simplicity, we stick to strict slicing. 
                # Ideally we want to keep endpoints.
                if (len(stroke)-1) not in subset_indices:
                     subset_indices.append(len(stroke)-1)
                
                # Sort just in case
                subset_indices = sorted(list(set(subset_indices)))
                
                x_sub = orig_x_full[subset_indices]
                y_sub = orig_y_full[subset_indices]
                
                # Parameterize
                t_sub = chord_length_parameterization(x_sub, y_sub)
                
                # Identify if closed curve (O)
                is_closed = (char_name == 'O')
                
                # Dictionary to hold fits for plotting
                fits_for_plot = {}
                
                # === Method 1: Cubic Spline (Natural) ===
                try:
                    cx, cy = fit_cubic_spline(t_sub, x_sub, y_sub, bc_mode='natural', is_closed=is_closed)
                    t_new = np.linspace(0, 1, 300)
                    fit_x = cx(t_new)
                    fit_y = cy(t_new)
                    
                    fits_for_plot['Cubic_Natural'] = (fit_x, fit_y)
                    
                    m = compute_metrics(fit_x, fit_y, reference_nodes)
                    results.append({
                        'Char': char_name, 'Stroke': s_idx+1,
                        'Downsample': d_factor, 'Method': 'Cubic_Natural',
                        'Mean_Err': m['mean_error'], 'Max_Err': m['max_error']
                    })
                except Exception as e:
                    print(f"    Error Cubic_Natural: {e}")

                # === Method 2: Cubic Spline (Clamped) ===
                # Not really applicable effectively if we don't know the derivative,
                # but our function estimates it from data.
                if not is_closed: # Clamped usually implies endpoints constraint
                    try:
                        cx, cy = fit_cubic_spline(t_sub, x_sub, y_sub, bc_mode='clamped', is_closed=False)
                        fit_x = cx(t_new)
                        fit_y = cy(t_new)
                        
                        fits_for_plot['Cubic_Clamped'] = (fit_x, fit_y)
                        
                        m = compute_metrics(fit_x, fit_y, reference_nodes)
                        results.append({
                            'Char': char_name, 'Stroke': s_idx+1,
                            'Downsample': d_factor, 'Method': 'Cubic_Clamped',
                            'Mean_Err': m['mean_error'], 'Max_Err': m['max_error']
                        })
                    except Exception as e:
                        print(f"    Error Cubic_Clamped: {e}")

                # === Method 3: B-Spline (Quadratic, k=2) ===
                try:
                    tck = fit_bspline(x_sub, y_sub, k=2, s=0, per=is_closed)
                    fit_x, fit_y = eval_bspline(tck, num_points=300)
                    
                    fits_for_plot['BSpline_Quad'] = (fit_x, fit_y)
                    
                    m = compute_metrics(fit_x, fit_y, reference_nodes)
                    results.append({
                        'Char': char_name, 'Stroke': s_idx+1,
                        'Downsample': d_factor, 'Method': 'BSpline_Quad',
                        'Mean_Err': m['mean_error'], 'Max_Err': m['max_error']
                    })
                except Exception as e:
                    print(f"    Error BSpline_Quad: {e}")
                    
                # === Method 4: B-Spline (Cubic, k=3) ===
                # Requires at least k+1 points. If downsample leaves too few, skip.
                if len(x_sub) > 3:
                    try:
                        tck = fit_bspline(x_sub, y_sub, k=3, s=0, per=is_closed)
                        fit_x, fit_y = eval_bspline(tck, num_points=300)
                        
                        fits_for_plot['BSpline_Cubic'] = (fit_x, fit_y)
                        
                        m = compute_metrics(fit_x, fit_y, reference_nodes)
                        results.append({
                            'Char': char_name, 'Stroke': s_idx+1,
                            'Downsample': d_factor, 'Method': 'BSpline_Cubic',
                            'Mean_Err': m['mean_error'], 'Max_Err': m['max_error']
                        })
                    except Exception as e:
                        print(f"    Error BSpline_Cubic: {e}")
                
                # === Plotting ===
                plot_filename = f"{char_name}_stroke{s_idx+1}_ds{d_factor}"
                plot_title = f"Char {char_name} (ds={d_factor}) Fit Comparison"
                
                plot_comparison(x_sub, y_sub, fits_for_plot, 
                                plot_title, plot_filename, plots_dir)
                                
    # Save CSV
    csv_path = os.path.join(tables_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['Char', 'Stroke', 'Downsample', 'Method', 'Mean_Err', 'Max_Err']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Done! metrics saved to {csv_path}")
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
