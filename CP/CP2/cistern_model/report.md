# Project Report: Cistern Filling Model

## Problem Statement
We model the filling of a toilet cistern using a float valve mechanism. This is an Initial Value Problem (IVP) for a system of two ODEs:

$$
\begin{aligned}
h'(t) &= \frac{q_{max} v(t) - k\sqrt{h(t)}}{A} \\
v'(t) &= \frac{v_{target}(h(t)) - v(t)}{\tau}
\end{aligned}
$$

where $v_{target}(h)$ is a smooth logistic function that closes the valve as $h \to h_{set}$.

## Numerical Approach
We use the **Backward Euler** implicit scheme for stability:
$$ u_{n+1} = u_n + dt \cdot f(t_{n+1}, u_{n+1}) $$

Since $f$ is nonlinear, we must solve for $u_{n+1}$ at each step. We implemented two solvers:

1.  **Fixed-Point Iteration**:
    - Rearranges the equation to $z = G(z)$.
    - Simple to implement but requires small $dt$ to converge fast.
    
2.  **Newton-Gauss-Seidel**:
    - Finds the root of $F(z) = z - u_n - dt \cdot f(t_{n+1}, z) = 0$.
    - Uses the Jacobian matrix $J_F = I - dt \cdot J_f$.
    - The linear system for the Newton update is solved using generic **Gauss-Seidel** iterations.

## Implementation Details
- **No SciPy**: We do not use `scipy.integrate` or `numpy.linalg.solve`. All integration and linear solving is manual.
- **Safety**: Square roots are protected ($\sqrt{h} \approx \sqrt{\max(h, \epsilon)}$) and variables are clamped to physical ranges ($h \ge 0$, $v \in [0,1]$).
- **Comparison**: We generated synthetic "measurements" with Gaussian noise to simulate a real experiment.

## Results & Findings
- **Convergence**: Newton's method generally requires fewer iterations per time step (quadratic convergence) compared to Fixed-Point (linear convergence), especially as $dt$ increases.
- **Stability**: Both methods compute the Implicit Euler step, so the solution trajectory is stable (A-stable method). However, the *solver* inside the step might fail. Fixed-point is more likely to diverge if $dt$ is too large (contractive mapping violation).
- **Accuracy**: Both methods converge to the same implicit solution (within tolerance). The error relative to a reference solution (small $dt$) decreases as $dt \to 0$.

## Conclusion
For this stiff-ish system with moderate time steps ($dt=0.5$ min), Newton-GS is more robust and efficient per step, though it has a higher per-iteration cost due to the Jacobian and linear solve. Fixed-Point is simpler but relies on $dt$ being small enough.
