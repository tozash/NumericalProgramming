# Demo Video Script: 3D Lemon Reconstruction
**Duration**: ~3:00

## [0:00 - 0:30] Introduction
**Visual**: Show the `input_lemon.jpg` on screen.
**Audio**: "Hello, this is [Your Name]. Today I will demonstrate Project 6.1: Reconstructing a 3D model of a lemon from a single 2D image. The goal is to apply numerical methods like differentiation and integration to analyze a real-world object."
**Action**: Briefly point to the lemon's curve and mention the assumption of axial symmetry.

## [0:30 - 1:00] Method: Edge & Axis Detection
**Visual**: Show `edges.png` and then `axis_overlay.png`.
**Audio**: "First, we convert the image to grayscale and compute gradients using finite difference approximations. This gives us the edge map you see here. Next, we determine the axis of symmetry. We iterate through possible vertical axes and calculate a symmetry score based on how well the reflected edges match the original ones. The red line indicates the detected center."

## [1:00 - 1:45] Profile Extraction & Curve Fitting
**Visual**: Show `fitted_profile.png` (scatter points vs red line).
**Audio**: "For every row in the image, we calculate the radius from the axis to the edge. The raw data is noisy, as shown by the blue dots. To fix this, we implement a parametric approximation. I used a cubic smoothing spline, which fits a smooth curve $r(y)$ through the data, minimizing the curvature energy. This gives us a mathematical definition of the lemon's profile."

## [1:45 - 2:30] 3D Reconstruction & Volume
**Visual**: Show `surface3d.png` (rotating if possible, or static).
**Audio**: "Using different angles theta from 0 to 360 degrees, we generate the 3D surface coordinates. Here is the resulting mesh. Finally, we compute the volume using the Disk Method. We numerically integrate $\pi r^2$ along the y-axis using the Trapezoidal Rule."
**Visual**: Show the terminal output with the calculated Volume.

## [2:30 - 3:00] Conclusion
**Visual**: Back to the side-by-side comparison of Input vs 3D Model.
**Audio**: "In conclusion, this project demonstrates how to go from raw pixels to a metrically useful 3D model using standard numerical algorithms. While the symmetry assumption is an approximation, the results are quite robust for typical produce. Thank you."
