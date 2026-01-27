import numpy as np
from tqdm import tqdm

def rk4_step(rhs, t, y, dt, *args):
    """
    Performs a single step of the Runge-Kutta 4 integration method.
    
    Args:
        rhs (callable): Function rhs(t, y, *args) returning dy/dt.
        t (float): Current time.
        y (np.ndarray): Current state.
        dt (float): Time step.
        *args: Additional arguments to pass to rhs.
        
    Returns:
        np.ndarray: Updated state at t + dt.
    """
    k1 = rhs(t, y, *args)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = rhs(t + dt, y + dt * k3, *args)
    
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_ivp_rk4(rhs, t_span, y0, dt, args=(), pbar=False):
    """
    Solves an initial value problem using the RK4 method with fixed time steps.
    
    Args:
        rhs (callable): Function rhs(t, y, *args) returning dy/dt.
        t_span (tuple): (t_start, t_end).
        y0 (np.ndarray): Initial state.
        dt (float): Time step.
        args (tuple): Additional arguments to pass to rhs.
        pbar (bool): Whether to show a progress bar.
        
    Returns:
        tuple: (times, states)
            - times: np.ndarray of shape (N_steps,)
            - states: np.ndarray of shape (N_steps, len(y0))
    """
    t_start, t_end = t_span
    num_steps = int(np.ceil((t_end - t_start) / dt))
    
    times = np.linspace(t_start, t_end, num_steps + 1)
    # Re-calculate actual dt to match t_end exactly if needed, 
    # but for fixed step usually we want constant dt.
    # Here we stick to the generated linspace times for consistency.
    actual_dt = times[1] - times[0]
    
    states = np.zeros((num_steps + 1, len(y0)))
    states[0] = y0
    
    current_y = y0
    
    iterator = range(num_steps)
    if pbar:
        iterator = tqdm(iterator, desc="Simulating")
        
    for i in iterator:
        t = times[i]
        current_y = rk4_step(rhs, t, current_y, actual_dt, *args)
        states[i + 1] = current_y
        
    return times, states
