import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show import config, utils
from drone_show.tasks import task3


def main():
    parser = argparse.ArgumentParser(description="Run Task 3: Swarm Following Tracked Centroid")
    parser.add_argument("--from-task2-npz", type=str, required=True,
                       help="Path to Task 2 trajectories.npz")
    
    # Centroid input options (mutually exclusive)
    centroid_group = parser.add_mutually_exclusive_group(required=True)
    centroid_group.add_argument("--centroids-csv", type=str,
                               help="Path to centroids.csv (pre-tracked)")
    centroid_group.add_argument("--video", type=str,
                               help="Path to video file for tracking")
    centroid_group.add_argument("--synthetic-video", action="store_true",
                               help="Use synthetic circular centroid path")
    
    # Video tracking options
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                       help="Bounding box for tracking (required if --video)")
    parser.add_argument("--select-roi", action="store_true",
                       help="Interactively select ROI from first frame")
    parser.add_argument("--stride", type=int, default=1,
                       help="Frame stride for tracking (default: 1)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to track")
    parser.add_argument("--min-features", type=int, default=30,
                       help="Minimum features for tracking (default: 30)")
    
    # Simulation parameters
    parser.add_argument("--dt", type=float, default=None,
                       help="Time step (default: 1/fps for video, 0.02 for synthetic)")
    parser.add_argument("--T", type=float, default=None,
                       help="Maximum duration (default: video duration or synthetic duration)")
    parser.add_argument("--output", type=str, default="outputs/task3",
                       help="Output directory (default: outputs/task3)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                       help="Random seed")
    parser.add_argument("--auto-params", action="store_true", default=True,
                       help="Auto-tune physics parameters (default: ON)")
    parser.add_argument("--no-auto-params", action="store_false", dest="auto_params",
                       help="Disable auto-tuning")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.video and not args.bbox and not args.select_roi:
        parser.error("--video requires --bbox or --select-roi")
    
    # Run Task 3
    try:
        task3.run_task3(
            from_task2_npz=args.from_task2_npz,
            centroids_csv=args.centroids_csv,
            video_path=args.video,
            synthetic_video=args.synthetic_video,
            bbox=tuple(args.bbox) if args.bbox else None,
            select_roi=args.select_roi,
            dt=args.dt,
            T=args.T,
            stride=args.stride,
            output_dir=args.output,
            seed=args.seed,
            auto_params=args.auto_params,
            max_frames=args.max_frames,
            min_features=args.min_features
        )
    except Exception as e:
        print(f"Task 3 Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
