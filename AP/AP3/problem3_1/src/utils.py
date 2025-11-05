"""
Utility functions for I/O operations: saving tables and figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def save_table_to_csv(data: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        filepath: Path to output CSV file (will create parent directories if needed)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Saved table to {filepath}")


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to output image file (will create parent directories if needed)
        dpi: Resolution in dots per inch (default: 300)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

