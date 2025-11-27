import math

def f(t, y):
    """
    Computes the RHS of the ODE: y' = 1/t^2 - y/t - y^2.
    """
    return 1.0/(t**2) - y/t - y**2

def exact_solution(t):
    """
    Returns the exact solution y(t) = -1/t.
    """
    return -1.0/t

def euler_explicit(f, a, b, y0, h):
    """
    Implements explicit Euler method:
        w_{i+1} = w_i + h * f(t_i, w_i)
    on [a, b] with step size h and initial value y(a) = y0.
    
    Returns:
        ts: list of t_i
        ws: list of w_i (Euler approximations)
    """
    # Number of steps. Using round() to avoid floating point issues with division
    N = int(round((b - a) / h))
    
    ts = [0.0] * (N + 1)
    ws = [0.0] * (N + 1)
    
    # Initialization
    ts[0] = a
    ws[0] = y0
    
    # Euler loop
    for i in range(N):
        # Update step using Euler formula
        ws[i+1] = ws[i] + h * f(ts[i], ws[i])
        ts[i+1] = ts[i] + h
        
    return ts, ws

def linear_interp(t0, y0, t1, y1, t):
    """
    Linear interpolation between (t0, y0) and (t1, y1):
        y(t) = y0 + (t - t0) * (y1 - y0) / (t1 - t0)
    """
    return y0 + (t - t0) * (y1 - y0) / (t1 - t0)

def interp_from_euler(ts, ws, t_query):
    """
    Locates the interval [t_i, t_{i+1}] containing t_query and
    returns the linearly interpolated value using linear_interp.
    """
    # Scan the grid to find the correct interval
    # We look for i such that ts[i] <= t_query <= ts[i+1]
    # Since ts is sorted, we can just iterate.
    for i in range(len(ts) - 1):
        if ts[i] <= t_query <= ts[i+1]:
            return linear_interp(ts[i], ws[i], ts[i+1], ws[i+1], t_query)
    
    # Fallback if t_query is slightly out of bounds due to float precision, 
    # though for this problem inputs are within range.
    # If t_query is effectively the last point:
    if abs(t_query - ts[-1]) < 1e-9:
        return ws[-1]
        
    raise ValueError(f"t_query={t_query} is out of bounds of the computed mesh.")

if __name__ == "__main__":
    # Parameters
    a = 1.0
    b = 2.0
    h = 0.05
    y0 = -1.0
    
    # Run Euler's method
    ts, ws = euler_explicit(f, a, b, y0, h)
    
    # Print grid results
    print(f"{'t_i':<10} {'w_i (Euler)':<15} {'y(t_i) (Exact)':<15} {'Error':<15}")
    print("-" * 55)
    for t_val, w_val in zip(ts, ws):
        y_exact = exact_solution(t_val)
        error = w_val - y_exact
        print(f"{t_val:<10.4f} {w_val:<15.6f} {y_exact:<15.6f} {error:<15.6e}")
        
    print("\n" + "="*55 + "\n")
    
    # Interpolation queries
    t_queries = [1.052, 1.555, 1.978]
    
    print(f"{'t_query':<10} {'Approx (Interp)':<20} {'Exact':<15} {'Error':<15}")
    print("-" * 60)
    
    for tq in t_queries:
        approx = interp_from_euler(ts, ws, tq)
        exact = exact_solution(tq)
        err = approx - exact
        print(f"{tq:<10.3f} {approx:<20.6f} {exact:<15.6f} {err:<15.6e}")

