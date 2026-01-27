import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show.tasks import task3_targets


def main():
    parser = argparse.ArgumentParser(description="Build Task 3 time-varying targets from tracked centroids")
    parser.add_argument("--from-task2-npz", type=str, required=True,
                       help="Path to Task 2 trajectories.npz")
    parser.add_argument("--centroids-csv", type=str, required=True,
                       help="Path to centroids.csv from video tracking")
    parser.add_argument("--bounds", type=float, nargs=4, default=[-1, 1, -1, 1],
                       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                       help="Simulation bounds for visualization (default: -1 1 -1 1)")
    parser.add_argument("--output", type=str, default="outputs/task3/debug_targets",
                       help="Output directory for debug files (default: outputs/task3/debug_targets)")
    
    args = parser.parse_args()
    
    from_task2_npz = Path(args.from_task2_npz)
    if not from_task2_npz.exists():
        print(f"Error: Task 2 NPZ file not found: {from_task2_npz}")
        sys.exit(1)
    
    centroids_csv = Path(args.centroids_csv)
    if not centroids_csv.exists():
        print(f"Error: Centroids CSV file not found: {centroids_csv}")
        sys.exit(1)
    
    print(f"Building Task 3 targets...")
    print(f"  Task 2 NPZ: {from_task2_npz}")
    print(f"  Centroids CSV: {centroids_csv}")
    print(f"  Output: {args.output}")
    
    try:
        target_fn, sample_T_series, P_ref, times_sec, centroids_sim = task3_targets.build_task3_targets(
            from_task2_npz=from_task2_npz,
            centroids_csv=centroids_csv,
            bounds=tuple(args.bounds),
            output_dir=args.output
        )
        
        print("\nSuccess!")
        print(f"  Reference formation: {len(P_ref)} drones")
        print(f"  Centroid samples: {len(times_sec)}")
        print(f"  Time range: {times_sec[0]:.2f} to {times_sec[-1]:.2f} seconds")
        print(f"\nDebug outputs saved to: {args.output}")
        print(f"  - targets_preview.png")
        print(f"  - T_series.npz")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
