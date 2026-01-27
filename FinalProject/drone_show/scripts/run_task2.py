import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show import config, utils
from drone_show.tasks import task2

def main():
    parser = argparse.ArgumentParser(description="Run Task 2: Transition Formation")
    parser.add_argument("--from-npz", type=str, required=True, help="Path to Task 1 trajectories.npz")
    parser.add_argument("--text", type=str, default="Happy New Year!", help="Target text for new formation")
    parser.add_argument("--dt", type=float, default=config.DEFAULT_DT, help="Time step")
    parser.add_argument("--duration", type=float, default=config.DEFAULT_T, help="Simulation duration")
    parser.add_argument("--output", type=str, default="outputs/task2", help="Output directory")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--sampling", type=str, default="fill", choices=["fill", "edge"], help="Sampling method")
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor for fill sampling")
    parser.add_argument("--auto-params", action="store_true", default=True, help="Auto-tune physics parameters")
    parser.add_argument("--no-auto-params", action="store_false", dest="auto_params", help="Disable auto-tuning")
    parser.add_argument("--T-trans", type=float, default=None, help="Transition duration (default: 0.8 * duration)")
    
    # Bounds parsing (optional, will be loaded from NPZ if available)
    parser.add_argument("--bounds", type=str, default=None, 
                       help="Bounds as 'xmin,xmax,ymin,ymax' (default: loaded from NPZ or (-4,4,-4,4))")
    
    args = parser.parse_args()
    
    # Parse bounds if provided
    bounds = None
    if args.bounds:
        try:
            parts = [float(x.strip()) for x in args.bounds.split(',')]
            if len(parts) == 4:
                bounds = tuple(parts)
            else:
                print(f"Warning: Invalid bounds format, using default")
        except ValueError:
            print(f"Warning: Could not parse bounds, using default")
    
    # Run Task 2
    try:
        task2.run_task2(
            from_npz=args.from_npz,
            text=args.text,
            duration=args.duration,
            dt=args.dt,
            output_dir=args.output,
            sampling=args.sampling,
            downsample=args.downsample,
            bounds=bounds,
            seed=args.seed,
            auto_params=args.auto_params,
            T_trans=args.T_trans
        )
    except Exception as e:
        print(f"Task 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
