import numpy as np

def initial_positions(N, mode="grid", bounds=(-1, 1, -1, 1)):
    """
    Generates initial positions for N drones.
    
    Args:
        N (int): Number of drones.
        mode (str): "grid", "line", or "random".
        bounds (tuple): (x_min, x_max, y_min, y_max).
        
    Returns:
        np.ndarray: Positions of shape (N, 2).
    """
    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    
    if mode == "grid":
        # Compute number of rows/cols for a roughly square aspect ratio
        # aspect = width / height
        # cols * rows >= N
        # cols / rows ~ aspect
        # rows^2 * aspect >= N  =>  rows >= sqrt(N/aspect)
        
        aspect = width / height if height > 0 else 1.0
        rows = int(np.ceil(np.sqrt(N / aspect)))
        cols = int(np.ceil(N / rows))
        
        # Ensure we have enough spots
        while rows * cols < N:
            rows += 1
            
        x_lin = np.linspace(xmin, xmax, cols)
        y_lin = np.linspace(ymin, ymax, rows)
        
        # Create grid
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_points = np.column_stack([xx.flatten(), yy.flatten()])
        
        # Select first N points
        # To make it centered/symmetric if N < rows*cols, we could be smarter,
        # but picking first N is sufficient for basic requirements.
        return grid_points[:N]
        
    elif mode == "line":
        # Line along X axis centered in Y
        x_lin = np.linspace(xmin, xmax, N)
        y_mid = (ymin + ymax) / 2.0
        y_lin = np.full(N, y_mid)
        return np.column_stack([x_lin, y_lin])
        
    elif mode == "random":
        # Uniform random within bounds
        # Seeding should be handled externally via utils.set_deterministic_behavior
        # or globally. But prompt says "OPTIONAL but must be seeded".
        # We assume global seed is set before calling this.
        xs = np.random.uniform(xmin, xmax, N)
        ys = np.random.uniform(ymin, ymax, N)
        return np.column_stack([xs, ys])
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
