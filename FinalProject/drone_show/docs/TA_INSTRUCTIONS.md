# TA Instructions (End-to-End Runs)

This document is a TA-focused running guide for Tasks 1–3, including required inputs, expected outputs, and common troubleshooting.

## Environment Setup

From the `drone_show` directory:

```bash
python -m venv .venv
```

Activate:
- Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

## Task 1 — Handwriting / Image → Static Formation

### Required input
- A handwriting/photo image file (e.g., `images/name.jpg`).

### Command

```bash
python scripts/run_task1.py \
  --image "images/name.jpg" \
  --n_drones 500 \
  --dt 0.02 \
  --duration 15 \
  --sampling fill \
  --shadow-correct \
  --output outputs/task1
```

### Expected outputs
Directory: `outputs/task1/`
- `trajectories.npz` (times, X, V, targets, params, bounds, errors)
- `summary.json`
- `analysis.json`
- `preview.png`
- `animation.mp4`
- debug images in `outputs/task1/debug/`:
  - `00_gray.png`, `01_corr.png`, `02_mask.png`, `03_edges.png`, `04_targets_only.png`

### Notes / tips
- If shadow edges dominate, keep `--shadow-correct` enabled and inspect `01_corr.png` and `02_mask.png`.
- For edge sampling (contours), use `--sampling edge` (edges are derived from the ink mask when shadow correction is enabled).

## Task 2 — Transition to “Happy New Year!”

### Required input
- Task 1 output: `outputs/task1/trajectories.npz`

### Command

```bash
python scripts/run_task2.py \
  --from-npz outputs/task1/trajectories.npz \
  --text "Happy New Year!" \
  --dt 0.1 \
  --duration 10 \
  --sampling fill \
  --auto-params \
  --output outputs/task2
```

### Expected outputs
Directory: `outputs/task2/`
- `trajectories.npz` (includes `T_series` for time-varying targets and final `targets`)
- `summary.json`
- `analysis.json`
- `preview.png`
- `animation.mp4`

## Task 3 — Video Tracking + Dynamic Formation

Task 3 can be run either by tracking a video in-process, or by first generating `centroids.csv` via the tracker and then running the swarm.

### Required inputs
- Task 2 output: `outputs/task2/trajectories.npz`
- A video file (e.g., `images/task3_video_easy.mp4`)
- An ROI (bounding box) to track (selected interactively or given as `--bbox x y w h`)

### Option A (recommended): track video first, then run Task 3 using CSV

#### 1) Track video → `centroids.csv`

```bash
python scripts/track_video.py \
  --video "images/task3_video_easy.mp4" \
  --select-roi \
  --output outputs/task3_video_tracking
```

Outputs in `outputs/task3_video_tracking/`:
- `centroids.csv`
- `first_frame.png`
- `tracked_path.png`
- `features_count.png`

#### 2) (Optional) Build and preview time-varying targets

```bash
python scripts/build_task3_targets.py \
  --from-task2-npz outputs/task2/trajectories.npz \
  --centroids-csv outputs/task3_video_tracking/centroids.csv \
  --output outputs/task3/debug_targets
```

Outputs in `outputs/task3/debug_targets/`:
- `targets_preview.png`
- `T_series.npz`

#### 3) Run Task 3 using the tracked centroids

```bash
python scripts/run_task3.py \
  --from-task2-npz outputs/task2/trajectories.npz \
  --centroids-csv outputs/task3_video_tracking/centroids.csv \
  --output outputs/task3_video
```

Outputs in `outputs/task3_video/`:
- `trajectories.npz` (times, X, V, P_ref, centroid series)
- `summary.json`
- `analysis.json`
- `animation.mp4`

### Option B: run Task 3 with synthetic centroid motion (no video required)

```bash
python scripts/run_task3.py \
  --from-task2-npz outputs/task2/trajectories.npz \
  --synthetic-video \
  --dt 0.02 \
  --T 6.0 \
  --output outputs/task3
```

## Troubleshooting

### ROI selection failed (“No ROI selected or invalid ROI”)
- Ensure you **drag a non-zero rectangle** and confirm with **Enter/Space** (ESC cancels).
- Try a tighter ROI around a high-contrast object.

### Tracking drift / low features
- Increase feature budget by using a tighter ROI and ensure the object has corners/texture.
- If you see many reseeds in `features_count.png`, the tracker is struggling.

### Animation MP4 not produced
Animations require an ffmpeg-backed writer. If MP4 fails:
- Install ffmpeg and ensure it’s on PATH.
- Alternatively, you can generate a GIF by choosing a `.gif` output path in visualization tools (where supported).

### dt too large → instability
- See the explicit failure demo: `python scripts/run_failure_cases.py` and review `outputs/failures/case1_dt_too_large/README.txt`.
- Typical stable dt values: `0.02`–`0.1` depending on duration and stiffness.

## Numerical Method Reference

See `docs/numerical_methods.md` for RK4 tableau, truncation error orders, and stability notes.

