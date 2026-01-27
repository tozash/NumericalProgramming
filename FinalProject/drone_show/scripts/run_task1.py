import argparse
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show import config, utils, geometry
from drone_show.preprocess import text_to_image
from drone_show.tasks import task1

def main():
    parser = argparse.ArgumentParser(description="Run Task 1: Static Formation")
    parser.add_argument("--image", type=str, help="Path to input image (handwriting)")
    parser.add_argument("--text", type=str, default="SANDRO", help="Text to generate if no image provided")
    parser.add_argument("--n_drones", type=int, default=config.DEFAULT_N_DRONES, help="Number of drones")
    parser.add_argument("--dt", type=float, default=config.DEFAULT_DT, help="Time step")
    parser.add_argument("--duration", type=float, default=config.DEFAULT_T, help="Simulation duration")
    parser.add_argument("--output", type=str, default="outputs/task1", help="Output directory")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--sampling", type=str, default="fill", choices=["fill", "edge"], help="Sampling method")
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor for fill sampling")
    parser.add_argument("--auto-params", action="store_true", default=True, help="Auto-tune physics parameters")
    parser.add_argument("--no-auto-params", action="store_false", dest="auto_params", help="Disable auto-tuning")
    
    # Shadow correction flags (for handwriting)
    parser.add_argument("--shadow-correct", action="store_true", default=None, help="Enable shadow correction (default: True for image input)")
    parser.add_argument("--no-shadow-correct", action="store_false", dest="shadow_correct", help="Disable shadow correction")
    parser.add_argument("--shadow-k-frac", type=float, default=0.12, help="Kernel size fraction for background estimation")
    parser.add_argument("--shadow-method", type=str, default="divide", choices=["divide", "subtract"], help="Illumination correction method")
    parser.add_argument("--thresh", type=str, default="adaptive", choices=["adaptive", "otsu"], help="Thresholding mode")
    parser.add_argument("--block-size", type=int, default=35, help="Block size for adaptive threshold (must be odd)")
    parser.add_argument("--C", type=float, default=10, help="Constant for adaptive threshold")
    parser.add_argument("--edge-from-mask", type=str, default="morph", choices=["morph", "canny"], help="Edge extraction method from mask")
    parser.add_argument("--canny-low", type=int, default=50, help="Low threshold for Canny (if edge-from-mask=canny)")
    parser.add_argument("--canny-high", type=int, default=150, help="High threshold for Canny (if edge-from-mask=canny)")
    
    args = parser.parse_args()
    
    # Default shadow_correct to True if image is provided
    if args.shadow_correct is None:
        args.shadow_correct = (args.image is not None)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = args.image
    if image_path is not None:
        # Try to resolve the path
        image_path = Path(image_path)
        if not image_path.is_absolute():
            # Try relative to current working directory
            if not image_path.exists():
                # Try relative to script directory
                script_dir = Path(__file__).resolve().parent.parent
                alt_path = script_dir / image_path
                if alt_path.exists():
                    image_path = alt_path
                else:
                    # Try relative to drone_show directory
                    drone_show_dir = script_dir / "drone_show"
                    alt_path = drone_show_dir / image_path
                    if alt_path.exists():
                        image_path = alt_path
        
        if not image_path.exists():
            print(f"Warning: Provided image path '{args.image}' not found. Falling back to text generation.")
            image_path = None
    
    if image_path is None or not image_path.exists():
        print(f"Generating image from text: '{args.text}'")
        img_arr = text_to_image(args.text, font_size=100, padding=40, thickness=5)
        image_path = debug_dir / "rendered_text.png"
        img_uint8 = (img_arr * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(image_path)
        print(f"Generated image saved to {image_path}")
    else:
        print(f"Using provided image: {image_path}")
    
    # Run Task
    try:
        task1.run_task1(
            image_path=image_path,
            n_drones=args.n_drones,
            duration=args.duration,
            dt=args.dt,
            output_dir=args.output,
            seed=args.seed,
            sampling=args.sampling,
            downsample=args.downsample,
            auto_params=args.auto_params,
            shadow_correct=args.shadow_correct,
            shadow_k_frac=args.shadow_k_frac,
            shadow_method=args.shadow_method,
            thresh_mode=args.thresh,
            thresh_block_size=args.block_size,
            thresh_C=args.C,
            edge_from_mask=args.edge_from_mask,
            canny_low=args.canny_low,
            canny_high=args.canny_high
        )
    except Exception as e:
        print(f"Task 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
