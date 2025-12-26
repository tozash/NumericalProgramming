# Parametric Spline Reconstruction of Characters T, O, Z

## 1. Introduction
This project investigates the reconstruction of digital character glyphs using parametric spline interpolation. We focus on three characters: **T** (consisting of two open strokes), **O** (a closed loop), and **Z** (a stroke with sharp corners). The goal is to correct fit smooth curves to these discrete point sets and analyze how different spline methods and node densities affect the reconstruction quality.

## 2. Data & Representation
The characters are defined as ordered sequences of 2D points $(x_i, y_i)$ within a $10 \times 10$ drawing box. Since planar curves like letters are multi-valued (e.g., an 'O' has two y-values for some x), we cannot use simple $y = f(x)$ interpolation. Instead, we use **parametric representation**:
$$ x = x(t), \quad y = y(t) $$
where $t$ is a parameter. We utilize **chord-length parameterization**, where $t_i$ is proportional to the cumulative Euclidean distance along the stroke, normalized to $t \in [0, 1]$. This prevents distortions that uniform parameterization might cause when nodes are unevenly spaced.

## 3. Methods

### 3.1 Cubic Spline Interpolation
A cubic spline consists of piecewise cubic polynomials $S_i(t)$ on each interval $[t_i, t_{i+1}]$ that join with $C^2$ continuity (continuous position, slope, and curvature). We compare three boundary conditions (BCs):
- **Natural Spline**: Sets the second derivative to zero at endpoints $S''(t_0) = S''(t_n) = 0$. This minimizes curvature but can flatten the curve at ends.
- **Clamped Spline**: Fixes the first derivative $S'(t_0)$ and $S'(t_n)$ to specific values. We estimate these slopes using finite distances from the first/last two points.
- **Periodic Spline**: Used for the closed character 'O', ensuring $S(t_0) = S(t_n)$, $S'(t_0) = S'(t_n)$, and $S''(t_0) = S''(t_n)$ for a seamless loop.

### 3.2 B-Splines (Basis Splines)
B-splines provide a flexible basis for curve representation. We use the `scipy.interpolate.splprep` routine to fit:
- **Quadratic B-Splines ($k=2$)**: Lower degree, less oscillatory, but $C^1$ continuity.
- **Cubic B-Splines ($k=3$)**: Standard choice for smooth curves ($C^2$).
We use $s=0$ to force the spline to interpolate (pass exactly through) the control points.

## 4. Experiments
We perform the following experiments for each character stroke:
1. **Node Downsampling**: We reconstruct the curves using:
   - All nodes (Baseline).
   - Every 2nd node (50% data).
   - Every 3rd node (33% data).
2. **Method Comparison**: We compare Natural Cubic, Clamped Cubic (where applicable), Quadratic B-spline, and Cubic B-spline.
3. **Metric**: We quantify error by densely sampling the fitted curve and computing the **average distance** from each *original* reference node to the nearest point on the fitted curve. This measures how well the curve adheres to the ground truth shape.

## 5. Results & Discussion

### Key Observations
- **Character T**: Being composed of straight lines, all methods perform well. However, cubic splines can exhibit "overshoot" (Runge's phenomenon-like artifacts) near the ends if node density is low.
- **Character O**: The Periodic Cubic Spline and Closed B-spline provide the best results, seamlessly closing the loop. Natural splines fail to close the loop smoothly, creating a discontinuity in slope at the join.
- **Character Z**: The sharp corners of 'Z' pose a challenge. Cubic splines try to smooth out the corners, resulting in "rounding" drift. Quadratic B-splines handle these tighter turns slightly better but still smooth them.

### Data Efficiency
Downsampling to every 3rd node significantly degrades the shape for 'O' and 'Z'. The 'O' becomes more polygonal or distorted, and the 'Z' loses its sharp definition.

## 6. Conclusion
- **Best General Method**: **Cubic B-splines** generally offer the most robust smooth interpolation for arbitrary shapes.
- **For Closed Loops ('O')**: Periodic boundary conditions are essential.
- **For Sharp Corners ('Z')**: Global splines struggle with local sharp features. A strictly piecewise linear approach or hybrid spline with variable knots would be better, but among smooth splines, higher density is required near corners.

**Comparison to Course Material**: Our results confirm the behavior described in *Sauer Numerical Analysis*: natural splines are "flatter" at boundaries, and parametric parameterization is critical for 2D curves.
