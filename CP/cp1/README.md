# Numerical Programming: Computational Project 1

This project implements a motion detection, tracking, and kinematic analysis pipeline for video data. It features two implementations:
1. **From Scratch**: Uses basic NumPy and manual algorithms for image processing and clustering.
2. **Library**: Uses OpenCV and Scikit-Learn for optimized performance.

## Installation

Ensure you have Python 3 installed. Install dependencies:

```bash
pip install numpy matplotlib opencv-python scipy scikit-learn
```

## Usage

**Important:** Run all commands from the `CP/cp1` directory (where this README is located).

### 1. Run Scratch Version
This uses manual background subtraction, finite difference derivatives, and custom K-Means clustering.

```bash
python -m src.pipeline_scratch --video data/video_one_object.mp4 --output results/one_object_scratch
```

### 2. Run Library Version
This uses OpenCV MOG2 background subtractor and Scikit-Learn KMeans.

```bash
python -m src.pipeline_lib --video data/video_multi_object.mp4 --output results/multi_object_lib
```

## Structure

- `src/derivatives.py`: Taylor-series finite difference approximations (1st-4th order).
- `src/clustering_scratch.py`: K-Means implementation with L1, L2, Linf, and Weighted norms.
- `src/background_model.py`: Manual vs OpenCV background modeling.
- `results/`: Contains generated plots (trajectories, kinematics) and clustering visualization.

