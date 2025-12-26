import numpy as np
import time
from model import f, jacobian_dfdu

def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=50):
    """
    Solves Ax = b using Gauss-Seidel method.
    x_i^(k+1) = (b_i - sum_{j<i} a_ij x_j^(k+1) - sum_{j>i} a_ij x_j^k) / a_ii
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
            
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
        
    return x

def fixed_point_step(t_next, u_n, dt, max_iter=50, tol=1e-8):
    """
    Solves z = u_n + dt * f(t_next, z) via Fixed-Point Iteration.
    """
    # Better initial guess: Explicit Euler predictor
    z = u_n + dt * f(t_next, u_n)  
    
    iters = 0
    converged = False
    
    for k in range(max_iter):
        iters += 1
        z_next = u_n + dt * f(t_next, z)
        
        # Check convergence
        if np.linalg.norm(z_next - z) < tol:
            z = z_next
            converged = True
            break
        
        z = z_next
        
    # Clamping for safety
    z[0] = max(z[0], 0.0)
    z[1] = np.clip(z[1], 0.0, 1.0)
    
    return z, iters, converged

def newton_gs_step(t_next, u_n, dt, max_iter=20, tol=1e-8, lin_tol=1e-10, lin_max_iter=50):
    """
    Solves F(z) = 0 via Newton's method, where F(z) = z - u_n - dt*f(t_next, z).
    Linear system J_F * delta = -F is solved by Gauss-Seidel.
    J_F = I - dt * df/du
    """
    # Better initial guess
    z = u_n + dt * f(t_next, u_n)
    
    iters = 0
    converged = False
    
    I = np.eye(len(u_n))
    
    for k in range(max_iter):
        iters += 1
        
        # F(z) = z - u_n - dt * f(z)
        F_val = z - u_n - dt * f(t_next, z)
        
        # Check convergence of F (residual)
        if np.linalg.norm(F_val) < tol:
            converged = True
            break
            
        # Jacobian of F: J_F = I - dt * J_f
        J_f = jacobian_dfdu(t_next, z)
        J_F = I - dt * J_f
        
        # Solve J_F * delta = -F_val using Gauss-Seidel
        # Use zeros as initial guess for delta update
        delta = gauss_seidel(J_F, -F_val, x0=np.zeros_like(F_val), tol=lin_tol, max_iter=lin_max_iter)

        # Update z
        z = z + delta
        
        # Check convergence of step size (optional but good)
        if np.linalg.norm(delta) < tol:
            converged = True
            break
            
    # Clamping
    z[0] = max(z[0], 0.0)
    z[1] = np.clip(z[1], 0.0, 1.0)

    return z, iters, converged

def backward_euler_integrate(u0, t_span, dt, method='newton_gs', **kwargs):
    """
    Integrates the ODE using Backward Euler with specified nonlinear solver.
    """
    t0, T = t_span
    N = int((T - t0) / dt)
    time_grid = np.linspace(t0, T, N + 1)
    
    u = np.zeros((N + 1, len(u0)))
    u[0] = u0
    
    u_curr = u0.copy()
    
    total_iters = 0
    iters_per_step = []
    fail_count = 0
    
    start_time = time.perf_counter()
    
    for n in range(N):
        t_next = time_grid[n+1]
        
        if method == 'nonlinear_fixed_point':
            u_next, iters, conv = fixed_point_step(t_next, u_curr, dt, **kwargs)
        elif method == 'newton_gs':
            u_next, iters, conv = newton_gs_step(t_next, u_curr, dt, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        if not conv:
            fail_count += 1
            # If not converged, accept result but warn (or we could reduce dt)
            # For this project we just count failures.
            
        u[n+1] = u_next
        u_curr = u_next
        
        total_iters += iters
        iters_per_step.append(iters)
        
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    stats = {
        'runtime': runtime,
        'total_iters': total_iters,
        'avg_iters': np.mean(iters_per_step),
        'fail_count': fail_count,
        'iters_history': np.array(iters_per_step)
    }
    
    return time_grid, u, stats
