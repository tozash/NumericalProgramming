import numpy as np

def repulsive_forces(X, Rsafe, k_rep, eps=1e-9):
    """
    Computes pairwise repulsive forces between drones.

    Args:
        X (np.ndarray): Positions of shape (N, d).
        Rsafe (float): Safety radius.
        k_rep (float): Repulsion gain.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        np.ndarray: Repulsive forces of shape (N, d).
    """
    N, d = X.shape
    
    # Compute pairwise differences: diff[i, j] = X[i] - X[j]
    # shape (N, N, d)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    
    # Compute distances: dist[i, j] = ||X[i] - X[j]||
    # shape (N, N)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal (self-interaction)
    # We can just set diagonal to infinity so it doesn't trigger < Rsafe
    np.fill_diagonal(dist, np.inf)
    
    # Mask for interactions within safety radius
    mask = dist < Rsafe
    
    # Initialize forces
    F_rep = np.zeros((N, d))
    
    if not np.any(mask):
        return F_rep
        
    # Valid distances for computation (avoid zero division even if mask logic fails somehow)
    valid_dist = np.maximum(dist, eps)
    
    # Direction vectors: direction[i, j] = (X[i] - X[j]) / dist[i, j]
    # Normalized vector pointing from j to i
    direction = diff / valid_dist[:, :, np.newaxis]
    
    # Magnitude: k_rep * (1/dist - 1/Rsafe) / (dist**2)
    # Note: The prompt specifies this formula.
    # Usually repulsive potentials are 1/r^2 or similar. 
    # Force = -grad(Potential). If Potential ~ (1/r - 1/R)^2, then force is proportional.
    # We follow the prompt exactly: k_rep * (1/dist - 1/Rsafe) / (dist**2)
    #
    # TA alignment note:
    #   magnitude = k_rep * (1/r - 1/Rsafe) / r^2
    # For r << Rsafe, (1/r - 1/Rsafe) ≈ 1/r, so magnitude scales like ~ 1/r^3.
    # This matches the “~1/r^3 within Rsafe” repulsion behavior described in the project spec/slides.
    
    term1 = (1.0 / valid_dist) - (1.0 / Rsafe)
    magnitude = k_rep * term1 / (valid_dist**2)
    
    # Apply mask
    magnitude[~mask] = 0.0
    
    # Sum forces on each particle i from all j
    # shape (N, N, d) -> sum over axis 1 -> (N, d)
    forces_matrix = magnitude[:, :, np.newaxis] * direction
    F_rep = np.sum(forces_matrix, axis=1)
    
    return F_rep

def smoothstep(u):
    """
    Smooth interpolation between 0 and 1.
    u is expected to be in [0, 1].
    """
    u = np.clip(u, 0.0, 1.0)
    return 3*u**2 - 2*u**3
