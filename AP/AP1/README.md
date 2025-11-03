# Vector and Matrix Norms: Implementation and Visualization

This project implements vector norms (1-norm and infinity norm) and their induced matrix norms, with visualization of unit ball slices.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy matplotlib pytest
```

## Usage

Run the main script to generate distances and unit ball visualizations:

```bash
python -m src.main --seed 7 --order row --norms 1,inf --savefigs paper/imgs/
```

### Command-line Arguments

- `--seed`: Random seed for reproducibility (default: 0)
- `--order`: Reshaping order for 2×2 matrices: "row" or "column" (default: "row")
- `--norms`: Comma-separated list of norms to use (default: "1,inf")
- `--savefigs`: Directory to save figure outputs (default: "paper/imgs/")

## Output

The script generates:
- Console output with computed distances (vector 1/∞ norm distances, matrix 1/∞ norm distances)
- Four PNG figures saved in the specified directory:
  - `vector_ball_norm1.png` - 2D slice of vector 1-norm unit ball
  - `vector_ball_norminf.png` - 2D slice of vector infinity-norm unit ball
  - `matrix_ball_norm1.png` - 2D slice of matrix 1-norm (induced) unit ball
  - `matrix_ball_norminf.png` - 2D slice of matrix infinity-norm (induced) unit ball

## Norms Implemented

- **Vector 1-norm**: $\|x\|_1 = \sum_i |x_i|$ (sum of absolute values)
- **Vector infinity-norm**: $\|x\|_\infty = \max_i |x_i|$ (maximum absolute value)
- **Matrix 1-norm (induced)**: $\|A\|_1 = \max_j \sum_i |a_{ij}|$ (maximum column sum)
- **Matrix infinity-norm (induced)**: $\|A\|_\infty = \max_i \sum_j |a_{ij}|$ (maximum row sum)

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── src/
│   ├── norms.py          # Norm implementations
│   ├── distances.py      # Distance calculations
│   ├── reshape.py        # Vector/matrix reshaping
│   ├── unit_ball.py      # Unit ball visualization
│   ├── utils.py          # Utility functions
│   └── main.py           # CLI entry point
├── tests/
│   └── test_norms.py     # Unit tests
└── paper/
    ├── paper.md          # 1-page explanation
    └── imgs/             # Generated figures
```

## Running Tests

```bash
pytest tests/ -v
```

