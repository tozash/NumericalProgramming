# Finite Difference Methods for Derivatives and Tangent Approximations

## Overview

This project implements finite difference methods to compute first derivatives of functions in one and two dimensions. We use only methods explicitly covered in class: central differences, two-point backward differences, and Richardson extrapolation. Additionally, we construct tangent lines (for 1D functions) and tangent planes with normal vectors (for 2D surfaces) using the computed derivatives.

In one dimension, given a function $f(x)$, we approximate its derivative $f'(x_0)$ at a point $x_0$ using finite difference stencils. The tangent line to the curve $y = f(x)$ at $x_0$ is given by $y = f(x_0) + f'(x_0)(x - x_0)$.

In two dimensions, for a surface $z = g(x, y)$, we compute partial derivatives $g_x$ and $g_y$ at a point $(x_0, y_0)$. The tangent plane is $z = g(x_0, y_0) + g_x(x - x_0) + g_y(y - y_0)$, and the normal vector is $\mathbf{n} = (-g_x, -g_y, 1)$.

We test the accuracy of these methods by comparing finite difference approximations to exact derivatives computed symbolically using SymPy. We perform accuracy sweeps over a geometric grid of step sizes $h$ and analyze how the error scales with $h$ to verify the theoretical order of convergence.

---

## Methods

### 1D Finite Difference Methods

We implement three methods for computing $f'(x_0)$:

**Central Difference (O($h^2$))**: The central difference formula (see `central_diff_1d` in `fd1d.py`) is:

$$f'(x_0) \approx \frac{f(x_0 + h) - f(x_0 - h)}{2h}$$

This uses symmetric points around $x_0$ and achieves second-order accuracy.

**Two-Point Backward Difference (O($h^2$))**: The backward difference formula (see `backward2_1d` in `fd1d.py`) is:

$$f'(x_0) \approx \frac{3f(x_0) - 4f(x_0 - h) + f(x_0 - 2h)}{2h}$$

This is useful when we cannot evaluate $f$ at $x_0 + h$ (e.g., at boundaries). It also achieves second-order accuracy.

**Richardson Extrapolation (O($h^4$))**: Richardson extrapolation (see `richardson_from_order2` in `fd1d.py`) boosts any O($h^2$) method to O($h^4$) by combining evaluations at two step sizes:

$$A \approx \frac{4D(f, x_0, h/2) - D(f, x_0, h)}{3}$$

where $D$ is an O($h^2$) derivative method. This works by canceling the leading $h^2$ error term. If $D(h) = f' + Ch^2 + O(h^4)$, then $D(h/2) = f' + C(h/2)^2 + O(h^4) = f' + Ch^2/4 + O(h^4)$. The combination $(4D(h/2) - D(h))/3 = f' + O(h^4)$ eliminates the $h^2$ term.

### 2D Finite Difference Methods

For a function $g(x, y)$, we compute partial derivatives using central differences (see `central_partial_x` and `central_partial_y` in `fd2d.py`):

$$g_x(x_0, y_0) \approx \frac{g(x_0 + h, y_0) - g(x_0 - h, y_0)}{2h}$$

$$g_y(x_0, y_0) \approx \frac{g(x_0, y_0 + h) - g(x_0, y_0 - h)}{2h}$$

The gradient is computed as $(g_x, g_y)$ (see `gradient_central` in `fd2d.py`).

**Tangent Plane**: The tangent plane to $z = g(x, y)$ at $(x_0, y_0)$ is constructed using the partial derivatives (see `tangent_plane` in `fd2d.py`):

$$z \approx g(x_0, y_0) + g_x(x - x_0) + g_y(y - y_0)$$

**Normal Vector**: The normal vector to the surface is (see `normal_vector` in `fd2d.py`):

$$\mathbf{n} = (-g_x, -g_y, 1)$$

**Optional: Mixed Derivative**: We also implement the mixed partial $g_{xy}$ using a 4-point central stencil (see `mixed_xy_central` in `fd2d.py`):

$$g_{xy}(x_0, y_0) \approx \frac{g(x_0+h, y_0+h) - g(x_0+h, y_0-h) - g(x_0-h, y_0+h) + g(x_0-h, y_0-h)}{4h^2}$$

### Test Functions

We test on:

- **1D**: $f_1(x) = e^x \cos x$ with exact derivative $f_1'(x) = e^x(\cos x - \sin x)$
- **2D**: $g_1(x, y) = x^2 y + y^2$ with exact partials $g_x = 2xy$ and $g_y = x^2 + 2y$

Exact derivatives are computed symbolically using SymPy (see `functions.py`) to avoid algebra mistakes.

### Experiment Design

We run accuracy sweeps over a geometric grid of step sizes: $h \in \{h_0, h_0/2, h_0/4, \ldots\}$. For each $h$, we compute the finite difference approximation and compare it to the exact derivative to get the absolute error. We then plot $\log(\text{error})$ vs $\log(h)$ and fit a line to estimate the order of convergence $p$ where $\text{error} \approx Ch^p$. The slope of this line should be approximately 2 for base methods and 4 for Richardson extrapolation (see `fit_order_slope` in `experiment.py`).

---

## Results

### 1D Derivative Accuracy

Table 1 (see `paper/tables/1d_errors.csv`) shows sample results for computing $f_1'(0.7)$ using different methods. The errors decrease as $h$ decreases, with Richardson extrapolation achieving the smallest errors.

Figure 1 (see `paper/figures/1d_error_vs_h.png`) shows log-log plots of error vs step size. The fitted slopes confirm:
- **Central difference**: order ≈ 2.0 (slope ≈ 2.0)
- **Backward difference**: order ≈ 2.0 (slope ≈ 2.0)
- **Richardson(central)**: order ≈ 4.0 (slope ≈ 4.0)

This matches the theoretical expectations. At very small $h$, rounding errors begin to dominate, causing the error to level off or increase slightly.

### 2D Partial Derivative Accuracy

Table 2 (see `paper/tables/2d_errors.csv`) shows results for computing $g_x(1.0, 3.0)$ and $g_y(1.0, 3.0)$ for $g_1(x, y) = x^2 y + y^2$. The exact values are $g_x = 6.0$ and $g_y = 7.0$.

Figures 2 and 3 (see `paper/figures/2d_g_x_error.png` and `paper/figures/2d_g_y_error.png`) show log-log error plots. Both partial derivatives exhibit second-order convergence (slope ≈ 2.0), as expected for central differences.

### Tangent Line and Plane

For the 1D function $f_1$ at $x_0 = 0.7$:
- **Exact tangent line**: $y = f(0.7) + f'(0.7)(x - 0.7)$
- **FD tangent line** (using central difference with smallest $h$): $y = f(0.7) + f'_{\text{FD}}(0.7)(x - 0.7)$

The FD-based tangent line coefficients match the exact values to high precision (typically 6-8 decimal places for the smallest $h$).

For the 2D function $g_1$ at $(x_0, y_0) = (1.0, 3.0)$:
- **Exact tangent plane**: $z = 12.0 + 6.0(x - 1.0) + 7.0(y - 3.0)$
- **FD tangent plane**: $z = 12.0 + g_{x,\text{FD}}(x - 1.0) + g_{y,\text{FD}}(y - 3.0)$

The FD-based tangent plane closely approximates the exact plane, with errors in the partial derivatives on the order of $10^{-6}$ to $10^{-8}$ for small $h$.

**Normal vector**: The exact normal is $\mathbf{n} = (-6.0, -7.0, 1.0)$, and the FD-computed normal matches closely.

### Summary of Fitted Orders

The summary file (see `paper/slopes_summary.txt`) reports all fitted slopes. All base methods (central, backward2) show slopes near 2.0, while Richardson extrapolation shows slopes near 4.0, confirming the theoretical orders.

---

## Discussion & Takeaways

### Rounding Error Limits

As $h$ becomes very small, rounding errors in floating-point arithmetic begin to dominate truncation errors. This is visible in the error plots as a "flattening" or upturn at the smallest $h$ values. The optimal $h$ balances truncation error (which decreases with $h$) and rounding error (which increases as $h$ decreases due to division by small numbers). For our test functions, this occurs around $h \approx 10^{-6}$ to $10^{-8}$.

### Richardson Extrapolation Benefits

Richardson extrapolation provides a significant accuracy improvement without requiring additional function evaluations beyond what's needed for the base method. By combining $D(h)$ and $D(h/2)$, we get an O($h^4$) result, which is especially valuable when function evaluations are expensive. The trade-off is slightly more computation, but the error reduction is substantial.

### 2D Considerations

Computing partial derivatives in 2D requires evaluating the function at four points (for central differences). The error behavior is similar to 1D: second-order convergence for central differences. The tangent plane and normal vector constructions are straightforward applications of the computed partials, providing a linear approximation to the surface near the evaluation point.

### Practical Notes

- Central differences are preferred when possible due to their symmetric nature and good accuracy.
- Backward differences are useful at boundaries or when forward points are unavailable.
- Richardson extrapolation is a powerful technique for improving accuracy without changing the base stencil.
- The order of convergence can be verified empirically by fitting slopes on log-log plots, which serves as a validation of the implementation.

All methods implemented here use only techniques explicitly covered in class, ensuring they align with the course material while providing robust numerical approximations to derivatives.

