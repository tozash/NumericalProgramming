import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

def set_dark_style(ax):
    """
    Configure axes with dark style: black background, faint grid, optional axis hiding.
    
    Args:
        ax: Matplotlib axes object.
    """
    ax.set_facecolor('black')
    ax.grid(True, alpha=0.2, color='white', linestyle='--')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

def plot_targets(ax, targets):
    """
    Plot target positions as red 'x' markers.
    
    Args:
        ax: Matplotlib axes object.
        targets (np.ndarray): Target positions of shape (N, 2).
    """
    if targets is not None and len(targets) > 0:
        ax.scatter(targets[:, 0], targets[:, 1], c='r', marker='x', s=50, 
                   linewidths=2, label='Targets', zorder=10)

def plot_drones(ax, X, label="Drones", color='b', alpha=0.6, size=20):
    """
    Plot drone positions as dots.
    
    Args:
        ax: Matplotlib axes object.
        X (np.ndarray): Drone positions of shape (N, 2).
        label (str): Label for legend.
        color (str): Color for dots.
        alpha (float): Transparency.
        size (float): Marker size.
    """
    if X is not None and len(X) > 0:
        ax.scatter(X[:, 0], X[:, 1], c=color, marker='o', s=size, 
                   alpha=alpha, label=label, zorder=5)

def plot_frame(ax, X=None, targets=None, title="", bounds=None, X_start=None, X_final=None):
    """
    Plot a single frame with drones, targets, and optional start/final positions.
    
    Args:
        ax: Matplotlib axes object.
        X (np.ndarray, optional): Current drone positions (N, 2). If None, only X_start/X_final used.
        targets (np.ndarray, optional): Target positions (N, 2).
        title (str): Plot title.
        bounds (tuple, optional): (xmin, xmax, ymin, ymax) to clamp view.
        X_start (np.ndarray, optional): Start positions (N, 2) to plot.
        X_final (np.ndarray, optional): Final positions (N, 2) to plot.
    """
    ax.clear()
    
    # Set bounds if provided
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
    # Plot targets first (background)
    if targets is not None:
        plot_targets(ax, targets)
    
    # Plot start positions (if provided)
    if X_start is not None:
        plot_drones(ax, X_start, label='Start', color='k', alpha=0.3, size=10)
    
    # Plot current/final positions
    if X_final is not None:
        plot_drones(ax, X_final, label='Drones (Final)', color='b', alpha=0.6, size=20)
    elif X is not None:
        plot_drones(ax, X, label='Drones', color='b', alpha=0.6, size=20)
    
    # Set title
    if title:
        ax.set_title(title, color='white' if ax.get_facecolor()[0] < 0.5 else 'black')
    
    # Maintain aspect ratio
    ax.set_aspect('equal')
    
    # Add legend if any labels exist
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

def animate_trajectories(npz_path, out_path, fps=30, trail=0, show_targets=True):
    """
    Create animation from saved trajectory NPZ file.
    
    Args:
        npz_path (str or Path): Path to trajectories.npz file.
        out_path (str or Path): Output path for animation (MP4 or GIF).
        fps (int): Frames per second.
        trail (int): Number of previous frames to show as trail (0 = no trail).
        show_targets (bool): Whether to show target positions.
        
    Returns:
        str: Path to created animation file (may differ from out_path if format changed).
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    
    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    
    times = data['times']
    X_series = data['X']  # Shape: (T, N, 2)
    
    T_steps, N, d = X_series.shape
    
    # Determine targets
    targets = None
    if 'T_series' in data:
        # Time-varying targets
        T_series = data['T_series']
        targets = T_series  # Shape: (T, N, 2)
    elif 'targets' in data:
        # Static targets
        targets_static = data['targets']
        # Broadcast to all time steps
        targets = np.tile(targets_static[np.newaxis, :, :], (T_steps, 1, 1))
    
    # Determine output format
    out_path_str = str(out_path)
    use_mp4 = out_path_str.endswith('.mp4')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine bounds from data
    all_x = X_series[:, :, 0].flatten()
    all_y = X_series[:, :, 1].flatten()
    if targets is not None:
        all_x = np.concatenate([all_x, targets[:, :, 0].flatten()])
        all_y = np.concatenate([all_y, targets[:, :, 1].flatten()])
    
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    x_margin = max(x_range * 0.1, 0.1) if x_range > 0 else 0.5
    y_margin = max(y_range * 0.1, 0.1) if y_range > 0 else 0.5
    bounds = (np.min(all_x) - x_margin, np.max(all_x) + x_margin,
              np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    
    # Animation update function
    def update(frame):
        ax.clear()
        ax.set_facecolor('white')  # Light background for animation
        ax.grid(True, alpha=0.3)
        
        # Current frame targets
        frame_targets = targets[frame] if targets is not None and show_targets else None
        
        # Plot targets
        if frame_targets is not None:
            plot_targets(ax, frame_targets)
        
        # Plot trail if requested
        if trail > 0:
            start_frame = max(0, frame - trail)
            for i in range(N):
                trail_x = X_series[start_frame:frame+1, i, 0]
                trail_y = X_series[start_frame:frame+1, i, 1]
                if len(trail_x) > 1:
                    # Fading alpha
                    n_trail = len(trail_x)
                    for j in range(n_trail - 1):
                        alpha = 0.1 + 0.9 * (j / max(1, n_trail - 1))
                        ax.plot(trail_x[j:j+2], trail_y[j:j+2], 'b-', alpha=alpha, linewidth=1)
        
        # Plot current positions
        plot_drones(ax, X_series[frame], label='', color='b', alpha=0.8, size=30)
        
        # Set bounds and aspect
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        
        # Title with time
        ax.set_title(f'Frame {frame}/{T_steps-1} (t={times[frame]:.2f}s)', fontsize=12)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=T_steps, interval=1000/fps, repeat=False)
    
    # Save animation
    try:
        if use_mp4:
            try:
                # Try MP4 with ffmpeg
                writer = 'ffmpeg'
                anim.save(str(out_path), writer=writer, fps=fps)
                print(f"Animation saved as MP4: {out_path}")
                return str(out_path)
            except Exception as e:
                print(f"MP4 save failed ({e}), falling back to GIF")
                use_mp4 = False
                out_path = out_path.with_suffix('.gif')
        
        if not use_mp4:
            # GIF fallback
            writer = PillowWriter(fps=fps)
            anim.save(str(out_path), writer=writer)
            print(f"Animation saved as GIF: {out_path}")
            return str(out_path)
            
    except Exception as e:
        print(f"Animation save failed: {e}")
        raise
    
    finally:
        plt.close(fig)
