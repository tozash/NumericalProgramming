import numpy as np
from . import forces

def speed_saturation(v, vmax):
    """
    Limits the magnitude of velocity vectors to vmax.
    
    Args:
        v (np.ndarray): Velocity vectors of shape (N, d).
        vmax (float): Maximum speed.
        
    Returns:
        np.ndarray: Saturated velocities of shape (N, d).
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    scale = np.minimum(1.0, vmax / (norm_v + 1e-12))
    return v * scale

def acceleration(X, V, T, params, t=0.0):
    """
    Computes acceleration for the drone swarm.
    
    Equation: vdot = (1/m) * (kp*(T - X) + alpha(t)*Frep - kd*V)
    
    Args:
        X (np.ndarray): Positions (N, d).
        V (np.ndarray): Velocities (N, d).
        T (np.ndarray): Target positions (N, d).
        params (dict): Dictionary containing parameters:
            - m: Mass
            - kp: Proportional gain
            - kd: Damping gain
            - k_rep: Repulsion gain
            - Rsafe: Safety radius
            - vmax: Max speed (not used directly in this formula but usually part of params)
            - total_time: Total simulation duration (optional, used for ramping)
        t (float): Current time.
            
    Returns:
        np.ndarray: Accelerations (N, d).
    """
    m = params['m']
    kp = params['kp']
    kd = params['kd']
    k_rep = params['k_rep']
    Rsafe = params['Rsafe']
    
    # Repulsive forces
    F_rep = forces.repulsive_forces(X, Rsafe, k_rep)
    
    # Repulsion ramping
    # alpha(t) = smoothstep(t / (0.4 * T_total))
    # If total_time is not in params, assume we are fully ramped up (alpha=1)
    if 'total_time' in params:
        T_total = params['total_time']
        if T_total > 0:
            u = t / (0.4 * T_total)
            alpha = forces.smoothstep(u)
        else:
            alpha = 1.0
    else:
        alpha = 1.0
        
    F_rep_scaled = alpha * F_rep
    
    # PID-like control force: F_control = kp * (Target - Position) - kd * Velocity
    F_control = kp * (T - X) - kd * V
    
    # Total force
    F_total = F_control + F_rep_scaled
    
    # Newton's law: a = F/m
    return F_total / m

def rhs(t, state, target_fn, params):
    """
    Right-hand side function for the ODE solver.
    
    Args:
        t (float): Current time.
        state (np.ndarray): Flattened state vector [X, V] of size 2*N*d.
        target_fn (callable): Function that takes t and returns target positions (N, d).
        params (dict): Parameters dictionary.
        
    Returns:
        np.ndarray: Flattened derivative [xdot, vdot] of size 2*N*d.
    """
    T = target_fn(t)
    N, d = T.shape
    
    expected_size = 2 * N * d
    if state.size != expected_size:
        raise ValueError(f"State size {state.size} does not match expected 2*N*d = {expected_size} for N={N}, d={d}")
        
    X = state[:N*d].reshape(N, d)
    V = state[N*d:].reshape(N, d)
    
    vmax = params['vmax']
    
    # xdot = Vsat
    Vsat = speed_saturation(V, vmax)
    xdot = Vsat
    
    # vdot = acceleration
    vdot = acceleration(X, V, T, params, t=t)
    
    # Flatten and concatenate
    return np.concatenate([xdot.flatten(), vdot.flatten()])
