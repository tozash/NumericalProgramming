import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show import video_tracking, video_io


def main():
    parser = argparse.ArgumentParser(description="Track object in video using optical flow")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                       help="Bounding box (x, y, width, height). If not provided, use --select-roi")
    parser.add_argument("--select-roi", action="store_true",
                       help="Interactively select ROI from first frame")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process")
    parser.add_argument("--stride", type=int, default=1,
                       help="Frame stride (1 = every frame, 2 = every other frame, etc.)")
    parser.add_argument("--bounds", type=float, nargs=4, default=[-1, 1, -1, 1],
                       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                       help="Simulation bounds for coordinate mapping (default: -1 1 -1 1)")
    parser.add_argument("--output", type=str, default="outputs/task3/debug_tracking",
                       help="Output directory for debug files")
    parser.add_argument("--min-features", type=int, default=30,
                       help="Minimum number of features to maintain (default: 30)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Get bounding box
    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.select_roi:
        print("Select ROI in the window (press SPACE or ENTER to confirm, ESC to cancel)")
        bbox = video_tracking.select_roi_first_frame(video_path)
        print(f"Selected ROI: {bbox}")
    else:
        print("Error: Must provide either --bbox or --select-roi")
        sys.exit(1)
    
    # Read video metadata
    fps, width, height, frame_count = video_io.read_video_meta(video_path)
    print(f"Video: {width}x{height}, {fps:.2f} fps, {frame_count} frames")
    print(f"Tracking with bbox: {bbox}")
    print(f"Max frames: {args.max_frames}, Stride: {args.stride}")
    
    # Track
    print("Tracking...")
    times_sec, centroids_px, status_info = video_tracking.track_centroid_optical_flow(
        video_path,
        init_bbox=bbox,
        max_frames=args.max_frames,
        stride=args.stride,
        min_features=args.min_features
    )
    
    print(f"Tracked {len(times_sec)} frames")
    print(f"Re-seeded {len(status_info['reseed_frames'])} times")
    
    # Map to simulation coordinates
    xmin, xmax, ymin, ymax = args.bounds
    centroids_sim = video_tracking.centroids_px_to_sim(
        centroids_px, width, height, bounds=(xmin, xmax, ymin, ymax)
    )
    
    # Save debug outputs
    output_dir = Path(args.output)
    print(f"Saving debug outputs to {output_dir}")
    video_tracking.save_debug_outputs(
        video_path,
        bbox,
        times_sec,
        centroids_px,
        centroids_sim,
        status_info,
        output_dir
    )
    
    print("Done!")
    print(f"Debug outputs saved to: {output_dir}")
    print(f"  - first_frame.png")
    print(f"  - tracked_path.png")
    print(f"  - features_count.png")
    print(f"  - centroids.csv")


if __name__ == "__main__":
    main()
