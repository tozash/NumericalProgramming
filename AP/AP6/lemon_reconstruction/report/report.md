# Numerical Programming - Project Report: 3D Lemon Reconstruction (Problem 6.1)

**Date**: 2025-12-20  
**Author**: sandro tozashvili

## 1. Problem Statement
The objective of this project is to reconstruct an approximate 3D model of a lemon from a single 2D side-view photograph. By assuming the lemon is approximately axially symmetric, we can model it as a surface of revolution. This involves detecting the object's edges, identifying the axis of symmetry, extracting the 2D profile curve, and fitting a parametric model (spline) to this profile. Finally, we compute the volume of the reconstructed object using numerical integration.

## 2. Model and Assumptions
We assume the lemon geometry can be approximated by rotating a profile curve $r(y)$ around a central vertical axis axis $x = x_0$.
- **Input**: A single 2D RGB image $I(x, y)$.
- **Output**: A 3D surface $S$ and volume $V$.
- **Assumption**: The lemon is strictly symmetric around a vertical axis. Irregularities in the real lemon are smoothed out or ignored.

## 3. Mathematical Methods

### 3.1 Edge Detection via Numerical Differentiation
To find the boundary of the lemon, we compute the image gradient $\nabla I = (I_x, I_y)$. We approximate the partial derivatives using central finite differences:
$$ I_x(x, y) \approx \frac{I(x+1, y) - I(x-1, y)}{2} $$
$$ I_y(x, y) \approx \frac{I(x, y+1) - I(x, y-1)}{2} $$
The gradient magnitude $M = \sqrt{I_x^2 + I_y^2}$ highlights regions of rapid intensity change (edges). We threshold $M$ to obtain a binary edge map.

### 3.2 Axis of Symmetry
We find the symmetry axis $x_0$ by minimizing a "mirror mismatch score". For a candidate axis $x$, we reflect the edge points across $x$ and measure their distance to the nearest actual edge points using a Euclidean Distance Transform (EDT). The $x$ that minimizes this total distance is chosen as the optimal axis.

### 3.3 Parametric Approximation (Spline)
The raw profile extracted from the image $r_{raw}(y)$ is often noisy. We fit a smoothing cubic spline to approximation $r(y)$. We use `scipy.interpolate.UnivariateSpline`, which minimizes:
$$ \sum (r_{raw}(y_i) - r(y_i))^2 + \lambda \int (r''(y))^2 dy $$
This balances fidelity to the data with smoothness of the curve.

### 3.4 Volume Computation
We use the **Disk Method** for finding the volume of a solid of revolution:
$$ V = \pi \int_{y_{min}}^{y_{max}} (r(y))^2 dy $$
Numerically, we approximate this integral using the **Trapezoidal Rule**:
$$ V \approx \pi \sum_{i=0}^{N-1} \frac{1}{2} (r(y_i)^2 + r(y_{i+1})^2) (y_{i+1} - y_i) $$

## 4. Experimental Setup
- **Input Image**: Smart phone photo of a lemon on a contrasting background.
- **Preprocessing**: Gaussian Blur ($5\times5$ kernel) to reduce noise.
- **Scale**: Results are computed in pixels unless a conversion factor (cm/pixel) is provided.

## 5. Results
The pipeline successfully generates:
1. **Edge Map**: Clearly delineates the lemon boundary.
2. **Axis**: Correctly identifies the vertical center.
3. **3D Surface**: A smooth 3D mesh resembling the original lemon.

*(See generated images in `assets/outputs/`)*

**Final Volume**: ~[Value from code] pixel$^3$.

## 6. Conclusions and Limitations
The method provides a robust first-order approximation of the lemon's shape.
**Limitations**:
- The axial symmetry assumption fails for irregular or curved lemons.
- Lighting shadows can cause false edges.
- Single view cannot capture depth deformations.

Overall, the project demonstrates the effective application of numerical differentiation, optimization (symmetry search), and integration.
