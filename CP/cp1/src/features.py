# cp1/src/features.py

import numpy as np
from .derivatives import compute_kinematics

def extract_features(track_data: dict, dt: float):
    """
    Extracts feature vector for clustering from a single object track.
    
    Args:
        track_data: Dictionary containing 'positions' (Nx2 array).
        dt: Time step.
        
    Returns:
        feature_vector: np.array [mean_speed, max_speed, mean_acc, max_acc, mean_jerk, mean_jounce]
    """
    positions = np.array(track_data['positions'])
    
    # Need sufficient length
    if len(positions) < 5:
        return np.zeros(6) # Not enough data
        
    # Separate X and Y for derivatives
    # Note: We compute derivatives on components, then take magnitude
    kinematics_x = compute_kinematics(positions[:, 0], dt)
    kinematics_y = compute_kinematics(positions[:, 1], dt)
    
    # Magnitudes
    # Speed = sqrt(vx^2 + vy^2)
    speed = np.sqrt(kinematics_x['velocity']**2 + kinematics_y['velocity']**2)
    
    # Acceleration Magnitude = sqrt(ax^2 + ay^2)
    accel = np.sqrt(kinematics_x['acceleration']**2 + kinematics_y['acceleration']**2)
    
    # Jerk Magnitude
    jerk = np.sqrt(kinematics_x['jerk']**2 + kinematics_y['jerk']**2)
    
    # Jounce Magnitude
    jounce = np.sqrt(kinematics_x['jounce']**2 + kinematics_y['jounce']**2)
    
    # Summary Stats
    mean_speed = np.mean(speed)
    max_speed = np.max(speed)
    
    mean_acc = np.mean(accel)
    max_acc = np.max(accel)
    
    mean_jerk = np.mean(jerk)
    mean_jounce = np.mean(jounce)
    
    return np.array([mean_speed, max_speed, mean_acc, max_acc, mean_jerk, mean_jounce])

def normalize_features(X):
    """
    Standardize features (z-score) before clustering so norms make sense.
    (x - mean) / std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8 # Avoid div by zero
    return (X - mean) / std

