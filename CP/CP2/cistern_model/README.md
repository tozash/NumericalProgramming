# Cistern Filling Model (ODE System)

This project models a toilet cistern / water tank filling system using a system of two Ordinary Differential Equations (ODEs). The system is solved using the **Implicit Backward Euler** method.

## Real-World Model
The system consists of a water tank with an automatic float valve.
- State variables:
  - $h(t)$: Water level [m]
  - $v(t)$: Valve opening fraction [0..1]
- The inflow is controlled by the valve, which closes as the water level approaches the target height $h_{set}$.
- The valve has a response delay $\tau$, making it a dynamic system rather than static feedback.

## Numerical Methods
We solve the discretized implicit equation $u_{n+1} = u_n + dt \cdot f(t_{n+1}, u_{n+1})$ using two different iterative solvers:

1.  **Fixed-Point Iteration**: A simple Picard iteration $u^{(k+1)} = u_n + dt \cdot f(t_{n+1}, u^{(k)})$.
2.  **Newton-Gauss-Seidel**: Newton's method is used to solve the nonlinear root-finding problem $F(z) = 0$. The linear update step $J\cdot\Delta = -F$ is solved using **Gauss-Seidel iterations** (no built-in linear algebra solver).

**Note:** No built-in ODE solvers (like `scipy.integrate.odeint`) were used. All methods were implemented manually.

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the simulation:
    ```bash
    python main.py --dt 0.5
    ```

    Options:
    - `--dt`: Time step in minutes (default 0.5).
    - `--T`: Total simulation time in minutes (default 30.0).
    - `--make-data`: Regenerate the synthetic noise measurements (saved to `data/measurements.csv`).

## Outputs
Results are saved in the `outputs/` folder:
- `trajectory_h.png`: Water level over time (comparing methods vs measurements).
- `trajectory_v.png`: Valve opening over time.
- `iterations_per_step.png`: Comparison of nonlinear iterations required by each method.
- `error_vs_reference.png`: Accuracy check against a high-precision reference run.
- `summary.txt`: Statistics including runtime, failure counts, and total iterations.
