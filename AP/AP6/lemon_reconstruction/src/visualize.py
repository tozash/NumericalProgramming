import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def plot_edges(edges, output_path):
    cv2.imwrite(output_path, edges)

def plot_axis(image, axis_x, output_path):
    """
    Overlays the detected axis on the original image.
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (axis_x, 0), (axis_x, h), (0, 0, 255), 2)
    cv2.imwrite(output_path, vis)

def plot_profile(y_points, r_points, y_fit, r_fit, output_path):
    plt.figure()
    # Invert y axis to match image coordinates
    plt.scatter(r_points, y_points, alpha=0.3, label='Extracted Points', s=1)
    
    # Sort for plotting line
    idx = np.argsort(y_fit)
    plt.plot(r_fit[idx], y_fit[idx], 'r-', linewidth=2, label='Fitted Model')
    
    plt.gca().invert_yaxis()
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Profile Approximation')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_3d_surface(y_fit, r_fit, output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    theta = np.linspace(0, 2*np.pi, 50)
    
    # y_fit and r_fit are 1D arrays
    # Create 2D mesh
    # We need to sort y first to ensure clean mesh
    idx = np.argsort(y_fit)
    y_sorted = y_fit[idx]
    r_sorted = r_fit[idx]
    
    # Take a subset of points to avoid too dense mesh
    step = max(1, len(y_sorted) // 100)
    y_plot = y_sorted[::step]
    r_plot = r_sorted[::step]
    
    Theta, Y = np.meshgrid(theta, y_plot)
    R = np.tile(r_plot[:, np.newaxis], (1, len(theta)))
    
    X = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Z')
    ax.set_title('Reconstructed 3D Surface of Lemon')
    
    # Make axis aspect ratio equal
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(output_path)
    plt.close()
