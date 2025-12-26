
import matplotlib.pyplot as plt
import os

def setup_plot_style():
    """Configures matplotlib style for cleaner plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6

def plot_single_fit(original_x, original_y, fit_x, fit_y, 
                    title, filename, output_dir):
    """
    Plots a single fit overlaying original nodes.
    
    Args:
        original_x, original_y: Original node coordinates.
        fit_x, fit_y: Fitted curve coordinates (dense).
        title: Plot title.
        filename: Output filename (without extension).
        output_dir: Directory to save points.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(original_x, original_y, 'ro', label='Nodes')
    plt.plot(fit_x, fit_y, 'b-', label='Fitted Curve')
    
    # Keep aspect ratio square to avoid distortion
    plt.axis('equal')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison(original_x, original_y, fits_dict, 
                    title, filename, output_dir):
    """
    Plots multiple fits on the same canvas.
    
    Args:
        original_x, original_y: Original node coordinates.
        fits_dict: Dictionary {method_name: (fit_x, fit_y)}.
        title: Plot title.
        filename: Output filename.
        output_dir: Directory.
    """
    plt.figure(figsize=(8, 8))
    plt.plot(original_x, original_y, 'ko', label='Original Nodes', zorder=10)
    
    # Cycle through some colors/styles
    colors = ['b', 'g', 'm', 'c', 'orange']
    styles = ['-', '--', '-.', ':']
    
    for i, (method, (fx, fy)) in enumerate(fits_dict.items()):
        c = colors[i % len(colors)]
        s = styles[i % len(styles)]
        plt.plot(fx, fy, color=c, linestyle=s, label=method, alpha=0.8)
        
    plt.axis('equal')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
