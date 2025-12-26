import argparse
import os
import sys
import numpy as np
import time
from model import f, jacobian_dfdu
from solvers import backward_euler_integrate
from plotting import plot_trajectory_h, plot_trajectory_v, plot_iterations, plot_error

def generate_measurements(output_path, dt_ref=0.05, T=30.0):
    """
    Generates synthetic measurement data using a high-precision reference run.
    """
    print("Generating synthetic measurements...")
    u0 = np.array([0.20, 1.00]) # h0, v0
    t_span = (0.0, T)
    
    # High precision run
    t_ref, u_ref, _ = backward_euler_integrate(u0, t_span, dt_ref, method='newton_gs', tol=1e-10, max_iter=50)
    
    # Sample every 2 minutes
    sample_indices = [i for i, t in enumerate(t_ref) if abs(t % 2.0) < dt_ref/2]
    
    t_sample = t_ref[sample_indices]
    u_sample = u_ref[sample_indices]
    
    # Add noise
    np.random.seed(42)
    h_noise = np.random.normal(0, 0.002, size=len(t_sample))
    v_noise = np.random.normal(0, 0.01, size=len(t_sample))
    
    h_meas = u_sample[:, 0] + h_noise
    v_meas = u_sample[:, 1] + v_noise
    
    # Clamp measurements to physical values roughly
    h_meas = np.maximum(h_meas, 0.0)
    v_meas = np.clip(v_meas, 0.0, 1.0)
    
    data = np.column_stack((t_sample, h_meas, v_meas))
    
    header = "t_min,h_m,v_open"
    np.savetxt(output_path, data, delimiter=',', header=header, comments='')
    print(f"Measurements saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Cistern Filling ODE Simulation (Problem 2.1)')
    parser.add_argument('--dt', type=float, default=0.5, help='Time step [min]')
    parser.add_argument('--T', type=float, default=30.0, help='Simulation time [min]')
    parser.add_argument('--make-data', action='store_true', help='Generate synthetic measurements')
    args = parser.parse_args()
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'outputs')
    data_dir = os.path.join(base_dir, 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    meas_file = os.path.join(data_dir, 'measurements.csv')
    
    # Generate data if requested or missing
    if args.make_data or not os.path.exists(meas_file):
        generate_measurements(meas_file, T=args.T)
        
    # Load measurements
    try:
        meas_data = np.loadtxt(meas_file, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Warning: Could not load measurements.csv: {e}")
        meas_data = None
        
    # Simulation Parameters
    u0 = np.array([0.20, 1.00]) # h=0.2m, v=1.0 (fully open)
    t_span = (0.0, args.T)
    
    print(f"Running Simulations with dt={args.dt} min...")
    
    # 1. Fixed-Point Iteration
    print("  -> Solver: Fixed-Point Iteration")
    t_fp, u_fp, stats_fp = backward_euler_integrate(
        u0, t_span, args.dt, method='nonlinear_fixed_point', 
        max_iter=50, tol=1e-8
    )
    
    # 2. Newton-Gauss-Seidel
    print("  -> Solver: Newton-Gauss-Seidel")
    t_ngs, u_ngs, stats_ngs = backward_euler_integrate(
        u0, t_span, args.dt, method='newton_gs', 
        max_iter=20, tol=1e-8, lin_tol=1e-10, lin_max_iter=50
    )
    
    # 3. Reference Solution for Error Analysis (Internal high-res run)
    print("  -> Generating reference solution for error analysis...")
    dt_ref = 0.05
    t_ref, u_ref, _ = backward_euler_integrate(u0, t_span, dt_ref, method='newton_gs', tol=1e-10)
    
    # --- Analysis & Plotting ---
    print("Generating plots and summary...")
    
    # Trajectories
    plot_trajectory_h(t_fp, u_fp, t_ngs, u_ngs, meas_data, output_dir)
    plot_trajectory_v(t_fp, u_fp, t_ngs, u_ngs, meas_data, output_dir)
    
    # Iterations
    plot_iterations(t_fp, stats_fp['iters_history'], stats_ngs['iters_history'], output_dir)
    
    # Errors
    max_err_h_fp, max_err_v_fp = plot_error(t_fp, u_fp, t_ref, u_ref, 'FixedPoint', output_dir)
    max_err_h_ngs, max_err_v_ngs = plot_error(t_ngs, u_ngs, t_ref, u_ref, 'NewtonGS', output_dir)
    
    # Summary Report
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Simulation Summary ===\n")
        f.write(f"Time Step dt: {args.dt} min\n")
        f.write(f"Total Time T: {args.T} min\n\n")
        
        f.write(f"Method: Fixed-Point Iteration\n")
        f.write(f"  Runtime: {stats_fp['runtime']:.4f} s\n")
        f.write(f"  Total Nonlinear Iters: {stats_fp['total_iters']}\n")
        f.write(f"  Avg Iters/Step: {stats_fp['avg_iters']:.2f}\n")
        f.write(f"  Failed Steps: {stats_fp['fail_count']}\n")
        f.write(f"  Max Error vs Ref (h): {max_err_h_fp:.2e}\n")
        f.write(f"  Max Error vs Ref (v): {max_err_v_fp:.2e}\n")
        f.write(f"  Final State: h={u_fp[-1,0]:.4f}, v={u_fp[-1,1]:.4f}\n\n")
        
        f.write(f"Method: Newton-Gauss-Seidel\n")
        f.write(f"  Runtime: {stats_ngs['runtime']:.4f} s\n")
        f.write(f"  Total Nonlinear Iters: {stats_ngs['total_iters']}\n")
        f.write(f"  Avg Iters/Step: {stats_ngs['avg_iters']:.2f}\n")
        f.write(f"  Failed Steps: {stats_ngs['fail_count']}\n")
        f.write(f"  Max Error vs Ref (h): {max_err_h_ngs:.2e}\n")
        f.write(f"  Max Error vs Ref (v): {max_err_v_ngs:.2e}\n")
        f.write(f"  Final State: h={u_ngs[-1,0]:.4f}, v={u_ngs[-1,1]:.4f}\n")
        
    print(f"Done! Results saved to {output_dir}/")
    print(open(summary_path).read())

if __name__ == "__main__":
    main()
