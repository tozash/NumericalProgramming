# Drone Show Simulation - Verification Report

## Mathematical Formulation (Explicit IVP)

We model each drone \(i\) with **position** \(x_i(t)\in\mathbb{R}^2\) and **velocity** \(v_i(t)\in\mathbb{R}^2\). Stacking all drones gives \(x(t)\in\mathbb{R}^{N\times 2}\) and \(v(t)\in\mathbb{R}^{N\times 2}\).

### State and dynamics (matches implementation)

Our implemented ODE (see `src/drone_show/dynamics.py`) is:

\[
\dot x = \mathrm{sat}(v, v_{\max})
\]

\[
\dot v = \frac{1}{m}\Big(k_p\,(T(t)-x) - k_d\,v + \alpha(t)\,F_{\mathrm{rep}}(x)\Big)
\]

Where:
- \(T(t)\) is the target formation (static for Task 1, smooth transition for Task 2, and centroid-driven motion for Task 3)
- \(F_{\mathrm{rep}}(x)\) is the pairwise repulsion term computed from inter-drone distances
- \(\alpha(t)\in[0,1]\) is a smooth ramp that gradually turns repulsion on (to avoid early clumping)

### Initial conditions and time domain

We solve an **Initial Value Problem (IVP)** on \(t\in[0,T]\) with:
- \(x(0)=x_0\) (grid/random initial positions depending on the task)
- \(v(0)=0\)

This project does **not** solve a boundary value problem (BVP): we do not specify boundary conditions like \(x(T)=x_T\) directly. Instead, we specify targets \(T(t)\) and solve the IVP forward in time.

> Numerical method details (RK4, truncation error, stability notes): see `docs/numerical_methods.md`.

## Linear vs Nonlinear Components

The system contains both linear and nonlinear parts:

- **Linear terms (per drone)**:
  - attraction/control toward targets: \(k_p\,(T(t)-x)\)
  - damping: \(-k_d\,v\)

- **Nonlinear coupling (all drones)**:
  - repulsion \(F_{\mathrm{rep}}(x)\) depends on **pairwise distances** between drones, so each drone’s acceleration depends on the positions of all nearby drones. This introduces nonlinear, coupled dynamics.

## Shooting Methods (Why Not Used)

Shooting methods are typically used to solve **boundary value problems** (BVPs), where conditions are specified at multiple times (e.g., \(x(0)\) and \(x(T)\)). Our simulation is formulated and solved as an **IVP**, so shooting is not required or used here.

## Task 3 “V(x,t)” Alignment Note

In Task 3, we track a centroid \(c(t)\) from video and move the target formation by **rigid translation**:

\[
T(t) = c(t) + \big(P_{\mathrm{ref}} - c_{\mathrm{ref}}\big),
\]

where \(P_{\mathrm{ref}}\) is the reference formation and \(c_{\mathrm{ref}}\) is its centroid. This translation-only motion implies a spatially-uniform velocity field:

\[
V(x,t) = \dot c(t),
\]

which can be approximated in practice by finite differences of the tracked centroids (the tracked positions are stored as a time series).

## Verification

### Unit Tests

The project includes comprehensive unit tests that guarantee correctness of individual components:

#### Core Components
- **`test_forces.py`**: Verifies force computation (attraction, repulsion, damping)
- **`test_solver.py`**: Validates RK4 integrator accuracy and stability
- **`test_assignment.py`**: Ensures Hungarian algorithm produces optimal assignments
- **`test_geometry.py`**: Tests shape extraction and point sampling algorithms

#### Task-Specific Tests
- **`test_task1_small.py`**: End-to-end Task 1 with convergence checks
- **`test_task1_converges.py`**: Verifies Task 1 convergence over time
- **`test_task2_small.py`**: End-to-end Task 2 transition with error checks
- **`test_task3_integration_synthetic.py`**: Full Task 3 pipeline with synthetic centroid
- **`test_task3_reproducible.py`**: Ensures deterministic behavior with same seed
- **`test_task3_targets_preserve_shape.py`**: Verifies rigid translation preserves pairwise distances

#### Video Tracking Tests
- **`test_video_tracking_synthetic.py`**: Optical flow tracking accuracy (< 5px error)
- **`test_centroids_mapping.py`**: Coordinate mapping correctness (Y-flip, bounds)

#### Image Processing Tests
- **`test_shadow_robust_mask.py`**: Shadow correction robustness
- **`test_handwriting_pipeline_debug_outputs.py`**: Debug output generation
- **`test_fill_sampling.py`**: Point sampling from masks
- **`test_extract_shape_points_multi_contour.py`**: Multi-contour extraction

#### Visualization Tests
- **`test_visualize_smoke.py`**: Animation generation
- **`test_visualize_uses_last_frame.py`**: Frame consistency

### Manual Checks

In addition to automated tests, the following manual checks are performed:

1. **Debug Images** (`outputs/task1/debug/`):
   - `00_gray.png`: Original grayscale image
   - `01_corr.png`: Illumination-corrected image
   - `02_mask.png`: Extracted ink mask
   - `03_edges.png`: Edge detection (if edge mode)
   - `04_targets_only.png`: Final target point distribution

2. **Tracking Debug** (`outputs/task3/debug_tracking/`):
   - `first_frame.png`: Initial frame with ROI bbox
   - `tracked_path.png`: Centroid path visualization
   - `features_count.png`: Feature count over time
   - `centroids.csv`: Tracked centroid data

3. **Target Preview** (`outputs/task3/debug_targets/`):
   - `targets_preview.png`: Reference formation + centroid path + snapshots
   - `T_series.npz`: Time-varying target positions

4. **Animation Review**:
   - Visual inspection of `animation.mp4` files for smooth motion
   - Check for collisions (drones overlapping)
   - Verify formation shape preservation

### Integration Tests

- **`test_run_all_smoke.py`**: Full pipeline smoke test (Task1 -> Task2 -> Task3)
- **`run_all.py`**: Complete pipeline execution with metric validation

## Failure Cases

The project includes explicit failure case demonstrations in `outputs/failures/`:

### Case 1: dt Too Large (`outputs/failures/case1_dt_too_large/`)
**Problem**: Using dt=1.0 causes numerical instability  
**Symptoms**: High final mean error, unstable trajectories  
**Root Cause**: Violates RK4 stability requirements  
**Fix**: Use smaller dt (0.02-0.1)

See `outputs/failures/case1_dt_too_large/README.txt` for details.

### Case 2: Rsafe Too Large (`outputs/failures/case2_Rsafe_too_large/`)
**Problem**: Rsafe=2.0 prevents convergence to tight formations  
**Symptoms**: Drones cannot reach targets due to excessive repulsion  
**Root Cause**: Rsafe larger than target spacing  
**Fix**: Use auto_params or set Rsafe < 0.6 * median_target_spacing

See `outputs/failures/case2_Rsafe_too_large/README.txt` for details.

### Case 3: Tracking Drift (`outputs/failures/case3_tracking_drift/`)
**Problem**: Low feature count (min_features=5) causes tracking failure  
**Symptoms**: In real video: tracking loses object, centroid drifts  
**Root Cause**: Insufficient features for robust optical flow  
**Fix**: Use higher min_features (default: 30), improve video quality

See `outputs/failures/case3_tracking_drift/README.txt` for details.

To regenerate failure cases:
```bash
python scripts/run_failure_cases.py
```

## Reproducibility

All simulations use deterministic random seeds for reproducibility. The default seed is 42 (configurable via `--seed`).

### Exact Commands

#### Full Pipeline
```bash
# Run complete pipeline: Task1 -> Task2 -> Task3
python scripts/run_all.py
```

#### Individual Tasks

**Task 1** (Static Formation):
```bash
python scripts/run_task1.py \
  --text "SANDRO" \
  --n_drones 100 \
  --dt 0.1 \
  --duration 10 \
  --sampling fill \
  --auto-params
```

**Task 2** (Transition):
```bash
python scripts/run_task2.py \
  --from-npz outputs/task1/trajectories.npz \
  --text "Happy New Year!" \
  --dt 0.1 \
  --duration 10 \
  --auto-params
```

**Task 3** (Following Centroid):
```bash
# With pre-tracked centroids
python scripts/run_task3.py \
  --from-task2-npz outputs/task2/trajectories.npz \
  --centroids-csv outputs/task3/debug_tracking/centroids.csv

# With synthetic path
python scripts/run_task3.py \
  --from-task2-npz outputs/task2/trajectories.npz \
  --synthetic-video \
  --dt 0.02 \
  --T 6.0
```

**Video Tracking**:
```bash
python scripts/track_video.py \
  --video path/to/video.mp4 \
  --select-roi \
  --output outputs/task3/debug_tracking
```

**Build Targets**:
```bash
python scripts/build_task3_targets.py \
  --from-task2-npz outputs/task2/trajectories.npz \
  --centroids-csv outputs/task3/debug_tracking/centroids.csv
```

### Reproducibility Verification

To verify reproducibility:
```bash
# Run same task twice with same seed
python scripts/run_task1.py --text "TEST" --seed 42 --output outputs/test1
python scripts/run_task1.py --text "TEST" --seed 42 --output outputs/test2

# Compare final positions (should be identical)
python -c "import numpy as np; d1=np.load('outputs/test1/trajectories.npz'); d2=np.load('outputs/test2/trajectories.npz'); print('Max diff:', np.max(np.abs(d1['X'][-1] - d2['X'][-1])))"
```

Expected output: `Max diff: 0.0` (or very close to 0, within numerical precision).

## AI Usage Statement

This project was developed with assistance from AI coding assistants (Cursor AI). The AI was used for:

1. **Code Generation**: Initial implementation of modules, functions, and test cases
2. **Code Review**: Identifying bugs, suggesting improvements, and ensuring best practices
3. **Documentation**: Generating docstrings, README content, and technical documentation
4. **Debugging**: Analyzing errors and proposing fixes
5. **Refactoring**: Improving code structure and organization

**Human Contributions**:
- Project requirements and specifications
- Design decisions and architecture choices
- Manual testing and validation
- Final review and approval of all code
- Integration of AI-generated code into the project

**AI-Generated Components**:
- Core simulation modules (`dynamics.py`, `solver.py`, `forces.py`)
- Task implementations (`task1.py`, `task2.py`, `task3.py`)
- Video tracking (`video_tracking.py`, `video_io.py`)
- Image processing (`preprocess.py`, `geometry.py`)
- Test suites (all files in `tests/`)
- CLI scripts (`scripts/*.py`)
- Analysis tools (`analysis.py`)

All code has been reviewed, tested, and validated by the human developer. The project follows standard software engineering practices with comprehensive testing and documentation.
