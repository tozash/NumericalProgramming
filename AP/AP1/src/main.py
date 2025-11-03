"""Main CLI entry point for the norms project."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .norms import norm1_vec, norminf_vec
from .distances import vector_distance, matrix_distance
from .reshape import vec4_to_mat2x2
from .unit_ball import plot_vector_unit_ball_slice, plot_matrix_unit_ball_slice
from .utils import two_random_vec4s


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Compute vector and matrix norms, distances, and visualize unit balls"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--order",
        type=str,
        default="row",
        choices=["row", "column"],
        help="Reshaping order for 2×2 matrices (default: row)",
    )
    parser.add_argument(
        "--norms",
        type=str,
        default="1,inf",
        help="Comma-separated list of norms (default: 1,inf)",
    )
    parser.add_argument(
        "--savefigs",
        type=str,
        default="paper/imgs/",
        help="Directory to save figures (default: paper/imgs/)",
    )

    args = parser.parse_args()

    # Parse norms
    norm_list = [n.strip() for n in args.norms.split(",")]
    for norm in norm_list:
        if norm not in ("1", "inf"):
            raise ValueError(f"Invalid norm: {norm}. Must be '1' or 'inf'.")

    # Create output directory
    output_dir = Path(args.savefigs)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate random vectors
    x, y = two_random_vec4s(seed=args.seed)
    print("=" * 60)
    print("Generated Vectors:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print()

    # Step 2: Reshape to 2×2 matrices
    X = vec4_to_mat2x2(x, order=args.order)
    Y = vec4_to_mat2x2(y, order=args.order)
    print("Reshaped to 2×2 Matrices:")
    print(f"  X =")
    print(f"    {X}")
    print(f"  Y =")
    print(f"    {Y}")
    print()

    # Step 3: Compute and print distances
    print("=" * 60)
    print("Distances:")
    print()
    print("Vector Distances:")
    for norm in norm_list:
        dist = vector_distance(x, y, norm)
        norm_name = "∞" if norm == "inf" else norm
        print(f"  ‖x - y‖_{norm_name} = {dist:.6f}")
    print()

    print("Matrix Distances (Induced Norms):")
    for norm in norm_list:
        dist = matrix_distance(X, Y, norm)
        norm_name = "∞" if norm == "inf" else norm
        print(f"  ‖X - Y‖_{norm_name} = {dist:.6f}")
    print("=" * 60)
    print()

    # Step 4: Choose references
    xr = x.copy()
    Xr = X.copy()

    # Step 5: Generate plots
    print(f"Generating plots and saving to {output_dir}...")

    for norm in norm_list:
        # Vector unit ball slice
        fig, ax = plot_vector_unit_ball_slice(
            norm_kind=norm, xr=xr, vary_idx=(0, 1), fixed=None
        )
        norm_name = "inf" if norm == "inf" else "1"
        filename = output_dir / f"vector_ball_norm{norm_name}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close(fig)

        # Matrix unit ball slice
        fig, ax = plot_matrix_unit_ball_slice(
            norm_kind=norm, Xr=Xr, vary_idx=(0, 1), order=args.order, fixed=None
        )
        filename = output_dir / f"matrix_ball_norm{norm_name}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()

