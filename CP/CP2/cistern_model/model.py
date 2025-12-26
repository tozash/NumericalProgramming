import numpy as np

# --- SYSTEM PARAMETERS ---
A = 0.20         # m^2, tank cross-section area
qmax = 0.010      # m^3/min, maximum inflow
k = 0.006         # m^3/(min*sqrt(m)), outflow coeff
tau = 1.0         # min, valve response time constant
h_set = 0.40      # m, desired fill level
delta = 0.03      # m, float sensitivity

# Small epsilon for safe sqrt
EPS_SQRT = 1e-12

def v_target(h):
    """
    Target valve opening logistic function:
    v_target(h) = 1 / (1 + exp((h - h_set) / delta))
    """
    # Safe exponent to avoid overflow if h is essentially infinite (though we clamp h)
    exponent = (h - h_set) / delta
    # If exponent is huge, exp is huge, return 0. If very negative, exp is 0, return 1.
    # Clip for safety just in case of divergence
    exponent = np.clip(exponent, -500, 500)
    return 1.0 / (1.0 + np.exp(exponent))

def dv_target_dh(h):
    """
    Derivative of v_target w.r.t h.
    dv_target/dh = -(1/delta) * v_target(h) * (1 - v_target(h))
    """
    vt = v_target(h)
    return -(1.0 / delta) * vt * (1.0 - vt)

def f(t, u):
    """
    ODE system right-hand side.
    u = [h, v]
    h' = (qmax * v - k * sqrt(h)) / A
    v' = (v_target(h) - v) / tau
    """
    h, v = u
    
    # Safe sqrt
    sqrt_h = np.sqrt(np.maximum(h, EPS_SQRT))
    
    dh_dt = (qmax * v - k * sqrt_h) / A
    dv_dt = (v_target(h) - v) / tau
    
    return np.array([dh_dt, dv_dt])

def jacobian_dfdu(t, u):
    """
    Jacobian matrix J = df/du evaluated at u.
    J = [[df1/dh, df1/dv],
         [df2/dh, df2/dv]]
    """
    h, v = u
    
    # df1/dh = -(k / (2 * A * sqrt(h)))
    sqrt_h = np.sqrt(np.maximum(h, EPS_SQRT))
    df1_dh = -(k) / (2.0 * A * sqrt_h)
    
    # df1/dv = qmax / A
    df1_dv = qmax / A
    
    # df2/dh = (1/tau) * dv_target/dh
    df2_dh = (1.0 / tau) * dv_target_dh(h)
    
    # df2/dv = -1/tau
    df2_dv = -1.0 / tau
    
    return np.array([[df1_dh, df1_dv],
                     [df2_dh, df2_dv]])
