# Video Script & Demo Plan (3 Minutes)

## Demo Checklist
1. Open Terminal in VS Code.
2. Ensure `metrics.csv` and `plots` folder are empty or show them being overwritten.
3. Run `python src/run_all.py`.
4. Show the printed output (looping through characters).
5. Open `outputs/plots` and cycle through `T_stroke1_ds1.png`, `O_stroke1_ds1.png`, `Z_stroke1_ds1.png`.
6. Open `outputs/tables/metrics.csv` to show numbers.

---

## Script

### 0:00 - 0:30 | Introduction
"Hello, this is my project on Parametric Spline Reconstruction for characters T, O, and Z. The goal is to take a set of discrete points representing these letters and fit smooth curves through them using different mathematical techniques. I'm using Python with SciPy for this analysis."

### 0:30 - 1:00 | Methods & Theory
"I implemented two main types of splines. First, **Cubic Splines**, where I tested 'Natural' boundary conditions (zero curvature at ends) and 'Clamped' conditions (fixed slope). For the letter 'O', I specifically used periodic boundaries to ensure a seamless closed loop. Second, I used **B-Splines** (both quadratic and cubic) via the `splprep` library, which naturally handles the parametric $x(t), y(t)$ form."

### 1:00 - 1:30 | The Implementation
"Here is the code structure. `param.py` handles the chord-length parameterization, ensuring the 'time' parameter $t$ scales with actual distance along the curve. `run_all.py` is the main driver that loops through downsampling levels (keeping every 2nd or 3rd node) to test how robust these methods are when data is scarce."

### 1:30 - 2:30 | Demo & Results
*(Run `python src/run_all.py` on screen)*
"I'll run the analysis now. You can see it processing T, O, and Z. It generates plots and a metrics table."

*(Open Plot for 'O')*
"Look at the 'O'. The Periodic fit (in cyan) closes perfectly, whereas a standard Natural spline might leave a gap or kink at the connection point. The metrics confirm that the error is minimal."

*(Open Plot for 'Z')*
"For 'Z', you can see the challenge of corners. Cubic splines tend to 'overshoot' or round off the sharp turns, especially when we remove nodes (downsampling). The B-spline approximates it well but still imposes smoothness where a sharp corner implies a discontinuity in derivative."

### 2:30 - 3:00 | Conclusion
"In conclusion, **Parametric Cubic B-splines** proved to be the most versatile tool, especially for closed loops. However, for shapes with sharp corners like 'Z', you either need high node density or splines that allow for $C^0$ continuity at specific knots. All code and the report are available in the submission. Thanks for watching!"
