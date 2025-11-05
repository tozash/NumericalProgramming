# Problem 3.1: Finite Difference Methods

This project implements 1D and 2D finite difference methods for computing derivatives, with Richardson extrapolation, tangent lines/planes, and normal vectors. All methods use only techniques explicitly covered in class: central differences, two-point backward differences, and Richardson extrapolation.

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:

   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
     If you get an execution policy error, run:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
     Then try activating again.
   
   - On Windows (Command Prompt):
     ```cmd
     .venv\Scripts\activate.bat
     ```
   
   - On Windows (Git Bash):
     ```bash
     source .venv/Scripts/activate
     ```
     Note: If `python` command is not found in Git Bash, use the full path:
     ```bash
     /c/Users/teona/AppData/Local/Programs/Python/Python313/python.exe -m src.cli ...
     ```
     Or use the Windows Python launcher:
     ```bash
     py -m src.cli ...
     ```
   
   - On macOS/Linux (bash):
     ```bash
     source .venv/bin/activate
     ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Full Experiment

To run both 1D and 2D experiments with default parameters:

```bash
python -m src.cli --func1d f1 --x0 0.7 --h0 0.5 --levels 6 \
                  --func2d g1 --x0-2d 1.0 --y0-2d 3.0 --h0-2d 0.5 --levels-2d 5 \
                  --out paper --with-mixed
```

### Command-Line Arguments

**1D Experiment:**
- `--func1d`: 1D function name (default: `f1` for exp(x)*cos(x))
- `--x0`: Evaluation point for 1D function (default: 0.7)
- `--h0`: Initial step size for 1D (default: 0.5)
- `--levels`: Number of h refinements for 1D (default: 6)

**2D Experiment:**
- `--func2d`: 2D function name (default: `g1` for x²y + y²)
- `--x0-2d`: x-coordinate for 2D function (default: 1.0)
- `--y0-2d`: y-coordinate for 2D function (default: 3.0)
- `--h0-2d`: Initial step size for 2D (default: 0.5)
- `--levels-2d`: Number of h refinements for 2D (default: 5)

**Output Options:**
- `--out`: Output directory base path (default: `paper`)
- `--with-mixed`: Enable computation of mixed derivative g_xy

### Example: Custom Parameters

```bash
python -m src.cli --func1d f1 --x0 1.0 --h0 0.1 --levels 8 \
                  --func2d g1 --x0-2d 2.0 --y0-2d 1.0 --h0-2d 0.1 --levels-2d 7 \
                  --out results
```

## Output Structure

After running the experiment, the following files are generated:

```
paper/
├── figures/
│   ├── 1d_error_vs_h.png          # 1D error vs step size (log-log)
│   ├── 2d_g_x_error.png           # 2D partial x error vs step size
│   ├── 2d_g_y_error.png           # 2D partial y error vs step size
│   └── 2d_g_xy_error.png          # Mixed derivative error (if --with-mixed)
├── tables/
│   ├── 1d_errors.csv               # 1D error table
│   └── 2d_errors.csv               # 2D error table
├── slopes_summary.txt              # Fitted order of convergence summary
└── paper.md                        # Paper document (see below)
```

### Tables

CSV tables contain columns:
- `method` (1D): Method name (central, backward2, richardson(central))
- `variable` (2D): Variable name (g_x, g_y, g_xy)
- `h`: Step size
- `value`: Computed derivative value
- `exact`: Exact derivative value
- `abs_error`: Absolute error

### Figures

Log-log plots show error vs step size with:
- Fitted slope annotations (expected ~2 for base methods, ~4 for Richardson)
- Legend for different methods/variables
- Grid for easy reading

## Viewing Results

1. **Open the paper**: View `paper/paper.md` in any Markdown viewer or editor.

2. **View figures**: Open PNG files in `paper/figures/` with any image viewer.

3. **Analyze tables**: Open CSV files in `paper/tables/` with Excel, pandas, or any spreadsheet application.

## Adding New Test Functions

To add a new 1D test function:

1. Edit `src/functions.py`
2. Add a new case in `get_1d_function()`:
   ```python
   elif name == 'f2':
       f_sym = <your SymPy expression>
       f_prime_sym = sp.diff(f_sym, x)
   ```

3. Use it via CLI: `--func1d f2`

To add a new 2D test function:

1. Edit `src/functions.py`
2. Add a new case in `get_2d_function()` and `get_2d_function_mixed()`:
   ```python
   elif name == 'g2':
       g_sym = <your SymPy expression>
       gx_sym = sp.diff(g_sym, x)
       gy_sym = sp.diff(g_sym, y)
   ```

3. Use it via CLI: `--func2d g2`

## Running Tests

Run the test suite to verify correctness and order of convergence:

```bash
pytest tests/ -v
```

Tests verify:
- Correctness on known polynomials
- Empirical order of convergence (~2 for base methods, ~4 for Richardson)
- Tangent plane and normal vector computations

## Project Structure

```
problem3_1/
├── src/
│   ├── __init__.py
│   ├── functions.py      # Test functions and exact derivatives
│   ├── fd1d.py           # 1D finite difference methods
│   ├── fd2d.py           # 2D partial derivatives, gradient, tangent plane, normal
│   ├── experiment.py     # Experiment runner, plotting, order fitting
│   ├── cli.py            # Command-line interface
│   └── utils.py          # I/O helpers
├── tests/
│   ├── __init__.py
│   ├── test_fd1d.py      # 1D method tests
│   └── test_fd2d.py      # 2D method tests
├── paper/
│   ├── figures/          # Generated plots
│   ├── tables/           # Generated CSV tables
│   └── paper.md          # Report document
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.11+
- numpy: Numerical computations
- matplotlib: Plotting
- sympy: Symbolic mathematics for exact derivatives
- pandas: Data tables and CSV export
- pytest: Testing framework (optional, for running tests)

## Notes

- All methods use pure NumPy (no SciPy derivative helpers, no automatic differentiation)
- Step sizes follow geometric progression: h, h/2, h/4, ...
- Order of convergence is estimated via linear regression on log-log plots
- Richardson extrapolation boosts O(h²) methods to O(h⁴) by canceling the leading error term

