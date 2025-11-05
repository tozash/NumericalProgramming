"""
Command-line interface for running finite difference experiments.
"""

import argparse
from .experiment import run_full_experiment


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run finite difference accuracy experiments for Problem 3.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --func1d f1 --x0 0.7 --h0 0.5 --levels 6 \\
                    --func2d g1 --x0-2d 1.0 --y0-2d 3.0 --h0-2d 0.5 --levels-2d 5 \\
                    --out paper --with-mixed
        """
    )
    
    # 1D experiment arguments
    parser.add_argument('--func1d', type=str, default='f1',
                       help='1D function name (default: f1)')
    parser.add_argument('--x0', type=float, default=0.7,
                       help='Evaluation point for 1D function (default: 0.7)')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial step size for 1D (default: 0.5)')
    parser.add_argument('--levels', type=int, default=6,
                       help='Number of h refinements for 1D (default: 6)')
    
    # 2D experiment arguments
    parser.add_argument('--func2d', type=str, default='g1',
                       help='2D function name (default: g1)')
    parser.add_argument('--x0-2d', type=float, default=1.0, dest='x0_2d',
                       help='x-coordinate for 2D function (default: 1.0)')
    parser.add_argument('--y0-2d', type=float, default=3.0, dest='y0_2d',
                       help='y-coordinate for 2D function (default: 3.0)')
    parser.add_argument('--h0-2d', type=float, default=0.5, dest='h0_2d',
                       help='Initial step size for 2D (default: 0.5)')
    parser.add_argument('--levels-2d', type=int, default=5, dest='levels_2d',
                       help='Number of h refinements for 2D (default: 5)')
    
    # Output options
    parser.add_argument('--out', type=str, default='paper',
                       help='Output directory base path (default: paper)')
    parser.add_argument('--with-mixed', action='store_true',
                       help='Compute mixed derivative g_xy')
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_full_experiment(
        func1d=args.func1d,
        x0_1d=args.x0,
        h0_1d=args.h0,
        levels_1d=args.levels,
        func2d=args.func2d,
        x0_2d=args.x0_2d,
        y0_2d=args.y0_2d,
        h0_2d=args.h0_2d,
        levels_2d=args.levels_2d,
        output_dir=args.out,
        with_mixed=args.with_mixed
    )
    
    print("\n" + "=" * 60)
    print("Experiments completed successfully!")
    print(f"Results saved to: {args.out}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

