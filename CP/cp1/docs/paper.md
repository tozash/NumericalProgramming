# Motion Analysis and Clustering via Numerical Derivatives

## 1. Introduction
This project explores the numerical analysis of motion from video data. The goal is to detect moving objects, reconstruct their kinematic properties (velocity, acceleration, jerk, jounce) using finite difference methods, and classify their motion patterns using clustering algorithms equipped with various norms.

## 2. Models & Methods

### 2.1 Motion Detection
We assume a static camera and background.
- **Scratch Approach**: We implemented a manual weighted grayscale conversion, followed by a 5x5 Box Blur (moving average convolution) to reduce noise. A running average background model ($B_t = \alpha I_t + (1-\alpha)B_{t-1}$) is maintained. The absolute difference $|I_t - B_t|$ is thresholded to detect foreground.
- **Library Approach**: Utilizes OpenCV's MOG2 (Gaussian Mixture-based Background/Foreground Segmentation), which is robust to lighting changes.

### 2.2 Numerical Differentiation
To analyze motion dynamics, we compute derivatives of the position $x(t)$ and $y(t)$. We use Taylor-series derived finite difference formulas:

1. **Velocity ($v$)**: Central Difference ($O(h^2)$)
   $$ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} $$
2. **Acceleration ($a$)**: Central Difference ($O(h^2)$)
   $$ f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2} $$
3. **Jerk ($j$) and Jounce ($s$)**: computed via higher-order central stencils (3rd and 4th derivatives).

Smoothing is critical before differentiation as derivatives amplify high-frequency noise. We apply a moving average filter ($N=5$) to the raw pixel coordinates.

### 2.3 Clustering and Norms
We construct feature vectors $F = [\bar{v}, v_{max}, \bar{a}, a_{max}, \bar{j}, \bar{s}]$ for each object. We group objects using K-Means with different distance metrics:
- **L2 (Euclidean)**: Standard geometric distance.
- **L1 (Manhattan)**: Robust to outliers in specific dimensions.
- **L-infinity (Chebyshev)**: Dominated by the largest single feature difference.
- **Weighted L2**: Assigns higher importance to jerk and jounce to distinguish erratic vs. smooth motion.

## 3. Experiments

We tested on two videos:
1. **Single Object**: A simple ball moving in a straight line.
2. **Multiple Objects**: Several objects moving with distinct speeds and irregularities.

### Results
- **Derivatives**: The raw finite differences were noisy. The moving average smoothing significantly clarified the acceleration and jerk signals.
- **Clustering**: The "Weighted L2" norm successfully separated "erratic" objects (high jerk) from "smooth" objects, whereas standard L2 was dominated by speed alone.

## 4. Conclusion
The "from-scratch" implementation highlights the sensitivity of numerical derivatives to noise and the importance of smoothing. While the library version (OpenCV) is faster and more robust to lighting, the manual implementation provides direct control over the numerical approximations, demonstrating the trade-offs between theoretical exactness and practical noise management.

