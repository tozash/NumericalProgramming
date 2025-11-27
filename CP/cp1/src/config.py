# cp1/src/config.py

"""
Configuration parameters for the project.
"""

# Video processing
RESIZE_WIDTH = 640  # Standardize processing width
RESIZE_HEIGHT = 480

# Background Subtraction (Scratch)
BG_ALPHA = 0.05        # Learning rate for running average background
BG_THRESHOLD = 30      # Absolute difference threshold
MIN_AREA = 50          # Minimum area for an object to be considered

# Smoothing
SMOOTHING_WINDOW = 5   # Window size for moving average (must be odd)
SPEED_THRESHOLD = 0.5  # Pixel/frame speed below which motion is zeroed

# Physical Units (Calibration placeholder)
# Example: 100 pixels = 1 meter
PIXELS_PER_METER = 100.0
FPS = 30.0             # Frames per second (assumed if not read from video)

# Finite Differences
TIME_STEP = 1.0 / FPS

# Clustering
NUM_CLUSTERS = 2
MAX_ITER = 100
# Weights for weighted norm: [mean_speed, max_speed, mean_acc, max_acc, mean_jerk, mean_jounce]
FEATURE_WEIGHTS = [1.0, 1.0, 2.0, 2.0, 5.0, 5.0] 

