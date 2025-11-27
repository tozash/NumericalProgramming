# Video Script: Numerical Motion Analysis

## 1. Intro (0:00 - 0:20)
"Hi, I'm [Name]. For my Numerical Programming project, I built a system to analyze motion in videos without using any deep learning. I wrote algorithms from scratch to detect objects, calculate their physics derivatives—like acceleration and jerk—and cluster them based on how they move."

## 2. Method Explanation (0:20 - 1:30)
"The core of the project is numerical differentiation. I track objects in pixel coordinates and use finite difference formulas—specifically Central Differences with order $O(h^2)$—to compute velocity, acceleration, and even higher-order terms like jerk and jounce.

Because taking derivatives amplifies noise, I implemented a custom moving average smoother.

For clustering, I wrote a K-Means algorithm that supports different norms. This lets me ask questions like: 'Are these objects similar because they have the same speed (L2 norm)? Or because they have the same peak jerk (Weighted norm)?' This allows me to distinguish between smooth and erratic motion."

## 3. Demo (1:30 - 2:30)
"Let me show you the code running.
First, I run the 'scratch' pipeline on a multi-object video:
`python -m src.pipeline_scratch --video data/video_multi_object.mp4`

(Show terminal output processing frames)

Now looking at the results:
- Here is the trajectory plot showing the paths of detected objects.
- This plot shows the velocity and acceleration. You can see how the smoothing removed the spikes from the raw data.
- Finally, this cluster plot shows how the Weighted Norm separated the erratic object from the smooth one, which the standard Euclidean distance missed."

## 4. Conclusion (2:30 - 3:00)
"In conclusion, this project demonstrates that classical numerical methods are powerful tools for computer vision. By carefully handling noise and choosing the right mathematical norms, we can extract meaningful physical insights from video data."

