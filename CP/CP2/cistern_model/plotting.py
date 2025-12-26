import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectory_h(t_fp, u_fp, t_ngs, u_ngs, meas_data, output_dir):
    """
    Plots water level h(t) for both methods vs measurements.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t_fp, u_fp[:, 0], 'b-', label='Fixed-Point', linewidth=2, alpha=0.7)
    plt.plot(t_ngs, u_ngs[:, 0], 'r--', label='Newton-GS', linewidth=2)
    
    if meas_data is not None:
        # t is col 0, h is col 1
        plt.scatter(meas_data[:, 0], meas_data[:, 1], c='k', marker='x', label='Measurements', zorder=5)
        
    plt.xlabel('Time [min]')
    plt.ylabel('Water Level h [m]')
    plt.title('Trajectory of Water Level h(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'trajectory_h.png'))
    plt.close()

def plot_trajectory_v(t_fp, u_fp, t_ngs, u_ngs, meas_data, output_dir):
    """
    Plots valve opening v(t).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t_fp, u_fp[:, 1], 'b-', label='Fixed-Point', linewidth=2, alpha=0.7)
    plt.plot(t_ngs, u_ngs[:, 1], 'r--', label='Newton-GS', linewidth=2)
    
    if meas_data is not None:
        # t is col 0, v is col 2
        plt.scatter(meas_data[:, 0], meas_data[:, 2], c='k', marker='x', label='Measurements', zorder=5)
        
    plt.xlabel('Time [min]')
    plt.ylabel('Valve Opening v [-]')
    plt.title('Trajectory of Valve Opening v(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'trajectory_v.png'))
    plt.close()

def plot_iterations(t, iters_fp, iters_ngs, output_dir):
    """
    Plots iterations per step.
    Note: t has length N+1, iters has length N. We plot against t[1:].
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t[1:], iters_fp, 'b-o', label='Fixed-Point', markersize=4, alpha=0.7)
    plt.plot(t[1:], iters_ngs, 'r-s', label='Newton-GS', markersize=4, alpha=0.7)
    
    plt.xlabel('Time [min]')
    plt.ylabel('Nonlinear Iterations')
    plt.title('Iterations per Time Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'iterations_per_step.png'))
    plt.close()

def plot_error(t, u_method, t_ref, u_ref, method_name, output_dir):
    """
    Computes and plots error vs reference. 
    Assumes t and t_ref start at same time but t_ref is finer.
    We interpolate ref to t.
    """
    # Interpolate reference solution to coarse time grid
    h_ref_interp = np.interp(t, t_ref, u_ref[:, 0])
    v_ref_interp = np.interp(t, t_ref, u_ref[:, 1])
    
    err_h = np.abs(u_method[:, 0] - h_ref_interp)
    err_v = np.abs(u_method[:, 1] - v_ref_interp)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(t, err_h, label='Error h', linewidth=2)
    plt.semilogy(t, err_v, label='Error v', linewidth=2, linestyle='--')
    
    plt.xlabel('Time [min]')
    plt.ylabel('Absolute Error')
    plt.title(f'Error vs Reference ({method_name})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(output_dir, f'error_vs_reference_{method_name}.png'))
    plt.close()
    
    return np.max(err_h), np.max(err_v)
