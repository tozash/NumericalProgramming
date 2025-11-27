# MiniQuiz 5.2: Euler's Method and Linear Interpolation

## 1. Project overview

This mini-project solves the initial value problem $y' = 1/t^2 - y/t - y^2$ on the interval $[1, 2]$ with $y(1) = -1$. It uses the Euler method with a step size of $h=0.05$ to approximate the solution. We then compare these approximations to the exact solution $y(t) = -1/t$. Finally, we use linear interpolation to estimate solution values at three specific points ($t=1.052$, $t=1.555$, and $t=1.978$) that lie between the mesh points.

## 2. Files in this mini-project

*   `miniquiz_5_2_euler.py` — Python implementation of Euler's method and interpolation. It contains the functions `f`, `exact_solution`, `euler_explicit`, `linear_interp`, and `interp_from_euler`.
*   `miniquiz_5_2_euler.m` — MATLAB implementation performing the exact same tasks with the same function names.
*   `README_miniquiz_5_2.md` — This document explaining the code and how to run it.

## 3. How to run (Python)

To run the Python version, open your terminal or command prompt, navigate to the folder containing the file, and type:

```bash
python miniquiz_5_2_euler.py
```

The script will print:
1.  A table showing the time steps $t_i$, the Euler approximation $w_i$ computed by `euler_explicit` (in `miniquiz_5_2_euler.py`), the exact value, and the error.
2.  A second table showing the interpolated approximations at the three query points computed by `interp_from_euler` (in `miniquiz_5_2_euler.py`), along with their exact values and errors.

## 4. How to run (MATLAB)

To run the MATLAB version:
1.  Open MATLAB and navigate to the directory containing `miniquiz_5_2_euler.m`.
2.  Type `miniquiz_5_2_euler` in the Command Window and press Enter.

The script will output the same two tables as the Python version. It uses the local function `euler_explicit` (in `miniquiz_5_2_euler.m`) for the steps and `interp_from_euler` (in `miniquiz_5_2_euler.m`) for the specific points.

## 5. What each function does

*   `f` (in both files) — Computes the right-hand side of the differential equation, $1/t^2 - y/t - y^2$.
*   `exact_solution` (in both files) — Returns the known exact solution $y(t) = -1/t$ for comparison.
*   `euler_explicit` (in both files) — Performs the Euler method loop $w_{i+1} = w_i + h f(t_i, w_i)$ to generate the mesh points.
*   `linear_interp` (in both files) — Applies the formula to interpolate linearly between two known points $(t_0, y_0)$ and $(t_1, y_1)$.
*   `interp_from_euler` (in both files) — Finds the correct interval in the Euler mesh that contains a query time $t$ and calls `linear_interp` to get the result.

