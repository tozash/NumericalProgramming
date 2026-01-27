import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from . import preprocess

def edges_from_image(img, method="canny", canny=(100, 200)):
    """Extracts binary edges from a grayscale image."""
    img_u8 = (img * 255).astype(np.uint8)
    if method == "canny":
        edges = cv2.Canny(img_u8, canny[0], canny[1])
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    return edges

def cv_contour_to_xy(contour, H):
    """Converts OpenCV contour to Cartesian XY coordinates with Y-flip."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 1] = (H - 1) - pts[:, 1]
    return pts

def contour_points(edges):
    """Extracts contours from a binary edge map."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    H, W = edges.shape
    result = []
    for cnt in contours:
        pts = cv_contour_to_xy(cnt, H)
        result.append(pts)
    return result

def sample_polyline_uniform(points, K):
    """Resamples a polyline to K points uniformly distributed by arc length."""
    if len(points) < 2:
        if len(points) == 1:
            return np.tile(points, (K, 1))
        return np.zeros((K, 2))
        
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.r_[0, np.cumsum(dists)]
    total_len = cum_dist[-1]
    
    if total_len == 0:
        return np.tile(points[0], (K, 1))
    
    target_dists = np.linspace(0, total_len, K)
    x_samp = np.interp(target_dists, cum_dist, points[:, 0])
    y_samp = np.interp(target_dists, cum_dist, points[:, 1])
    
    return np.column_stack([x_samp, y_samp])

def smooth_contour_spline(points, K):
    """Fits a cubic spline to the contour and resamples K points."""
    if len(points) < 4:
        return sample_polyline_uniform(points, K)
    
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.r_[0, np.cumsum(dists)]
    total_len = cum_dist[-1]
    
    if total_len == 0:
        return np.tile(points[0], (K, 1))
        
    is_closed = np.linalg.norm(points[0] - points[-1]) < 1e-3 * total_len
    t = cum_dist / total_len
    bc = 'periodic' if is_closed else 'not-a-knot'
    
    unique_mask = np.r_[True, dists > 1e-6]
    if np.sum(unique_mask) < 4:
         return sample_polyline_uniform(points, K)
         
    t_clean = t[unique_mask]
    pts_clean = points[unique_mask].copy()
    
    if is_closed and bc == 'periodic':
        pts_clean[-1] = pts_clean[0]
    
    cs_x = CubicSpline(t_clean, pts_clean[:, 0], bc_type=bc)
    cs_y = CubicSpline(t_clean, pts_clean[:, 1], bc_type=bc)
    
    t_new = np.linspace(0, 1, K)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    
    return np.column_stack([x_new, y_new])

def sample_points_from_mask(mask, K, downsample=2, method="farthest"):
    """
    Samples K points from a binary mask using farthest-point sampling to ensure coverage.
    
    Args:
        mask (np.ndarray): uint8 mask {0, 1}.
        K (int): Number of points.
        downsample (int): Grid step for candidate points (optimization).
        method (str): "farthest" (default) or "random".
        
    Returns:
        np.ndarray: Points (K, 2) in image coordinates (x, y_flipped).
    """
    H, W = mask.shape
    
    # Get candidate pixels (y, x) where mask is 1
    # Downsample for speed
    ys, xs = np.where(mask[::downsample, ::downsample] > 0)
    
    if len(ys) == 0:
        # No foreground
        return np.zeros((K, 2))
        
    # Scale back to original coordinates
    # Add offset to center in the block
    ys = ys * downsample + downsample // 2
    xs = xs * downsample + downsample // 2
    
    candidates = np.column_stack([xs, ys]).astype(np.float32)
    N_cand = len(candidates)
    
    if N_cand <= K:
        # Not enough candidates, return all and duplicate some
        result = candidates
        if N_cand < K:
            # Pad with random choices from candidates
            pad_indices = np.random.choice(N_cand, K - N_cand)
            result = np.vstack([result, candidates[pad_indices]])
    else:
        # Downselect K points
        if method == "random":
            indices = np.random.choice(N_cand, K, replace=False)
            result = candidates[indices]
        elif method == "farthest":
            # Farthest point sampling
            # Start with point closest to centroid
            centroid = np.mean(candidates, axis=0)
            dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
            start_idx = np.argmin(dists_to_centroid)
            
            selected_indices = [start_idx]
            
            # Distance from each candidate to the SET of selected points
            # Initialize with dist to first point
            min_dists = np.linalg.norm(candidates - candidates[start_idx], axis=1)
            
            for _ in range(1, K):
                # Pick point with max min_dist
                next_idx = np.argmax(min_dists)
                selected_indices.append(next_idx)
                
                # Update min_dists
                new_dists = np.linalg.norm(candidates - candidates[next_idx], axis=1)
                min_dists = np.minimum(min_dists, new_dists)
                
            result = candidates[selected_indices]
    
    # Flip Y to Cartesian coordinates
    result[:, 1] = (H - 1) - result[:, 1]
    
    return result

def extract_shape_points_from_image(path, K, smooth=True, min_area_ratio=0.0005, debug_callback=None, sampling="fill", downsample=2,
                                    shadow_correct=False, shadow_k_frac=0.12, shadow_method="divide",
                                    thresh_mode="adaptive", thresh_block_size=35, thresh_C=10,
                                    edge_from_mask="morph", canny_low=50, canny_high=150):
    """
    Extracts shape points using either contour edges or mask fill.
    Supports shadow correction for handwriting images.
    
    Args:
        path (str): Path to image.
        K (int): Number of points.
        smooth (bool): Only for 'edge' mode.
        min_area_ratio (float): Only for 'edge' mode.
        debug_callback (callable): Optional callback(artifact_dict).
        sampling (str): "fill" or "edge".
        downsample (int): Only for 'fill' mode.
        shadow_correct (bool): Enable shadow/illumination correction.
        shadow_k_frac (float): Kernel size fraction for background estimation.
        shadow_method (str): "divide" or "subtract" for illumination correction.
        thresh_mode (str): "adaptive" or "otsu" for thresholding.
        thresh_block_size (int): Block size for adaptive threshold.
        thresh_C (float): Constant for adaptive threshold.
        edge_from_mask (str): "morph" or "canny" for edge extraction from mask.
        canny_low (int): Low threshold for Canny.
        canny_high (int): High threshold for Canny.
        
    Returns:
        np.ndarray: Points (K, 2).
    """
    
    # Load grayscale image
    gray = preprocess.load_image_gray(path)
    H, W = gray.shape
    
    # Shadow correction pipeline (for handwriting)
    if shadow_correct:
        # Convert to uint8 for processing
        gray_u8 = (gray * 255).astype(np.uint8)
        
        # Step 1: Illumination correction
        corr_u8 = preprocess.illumination_correct(gray, method=shadow_method, k_frac=shadow_k_frac)
        
        # Step 2: Extract ink mask
        mask_u8 = preprocess.ink_mask_from_corrected(corr_u8, mode=thresh_mode, 
                                                     block_size=thresh_block_size, C=thresh_C)
        
        # Debug outputs
        if debug_callback:
            debug_callback({
                'gray': gray_u8,
                'corr': corr_u8,
                'mask': mask_u8
            })
        
        if sampling == "fill":
            # Fill mode: sample directly from mask
            mask = (mask_u8 > 127).astype(np.uint8)
            points = sample_points_from_mask(mask, K, downsample=downsample)
            return points
            
        elif sampling == "edge":
            # Edge mode: extract edges from mask (not from raw image)
            edges = preprocess.edges_from_mask(mask_u8, method=edge_from_mask, 
                                               canny_low=canny_low, canny_high=canny_high)
            
            if debug_callback:
                debug_callback({'edges': edges})
            
            raw_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            img_area = H * W
            valid_contours = []
            for cnt in raw_contours:
                area = cv2.contourArea(cnt)
                if area >= min_area_ratio * img_area:
                    pts = cv_contour_to_xy(cnt, H)
                    valid_contours.append(pts)
                    
            if not valid_contours:
                raise ValueError("No valid contours found in image after filtering")
                
            if debug_callback:
                debug_callback({'contours': valid_contours})

            lengths = []
            for pts in valid_contours:
                if len(pts) < 2:
                    lengths.append(0)
                else:
                    d = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                    lengths.append(d)
                    
            total_len = sum(lengths)
            if total_len == 0:
                return np.zeros((K, 2))
                
            counts = []
            for l in lengths:
                c = int(np.round(K * l / total_len))
                counts.append(c)
                
            diff = K - sum(counts)
            if diff != 0:
                idx_max = np.argmax(lengths)
                counts[idx_max] += diff
                
            all_samples = []
            for i, pts in enumerate(valid_contours):
                c = counts[i]
                if c <= 0: continue
                    
                if smooth:
                    sampled = smooth_contour_spline(pts, c)
                else:
                    sampled = sample_polyline_uniform(pts, c)
                all_samples.append(sampled)
                
            if not all_samples:
                 return np.zeros((K, 2))
                 
            result = np.vstack(all_samples)
            
            if len(result) > K:
                result = result[:K]
            elif len(result) < K:
                pad = np.tile(result[-1], (K - len(result), 1))
                result = np.vstack([result, pad])
                
            return result
        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")
    
    else:
        # Original pipeline (no shadow correction)
        if sampling == "fill":
            # Fill mode (dot matrix style)
            mask = preprocess.to_binary_mask_from_image(path)
            
            if debug_callback:
                debug_callback({'mask': mask})
                
            points = sample_points_from_mask(mask, K, downsample=downsample)
            return points
            
        elif sampling == "edge":
            # Edge/Contour mode
            edges = edges_from_image(gray)
            
            raw_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            img_area = H * W
            valid_contours = []
            for cnt in raw_contours:
                area = cv2.contourArea(cnt)
                if area >= min_area_ratio * img_area:
                    pts = cv_contour_to_xy(cnt, H)
                    valid_contours.append(pts)
                    
            if not valid_contours:
                raise ValueError("No valid contours found in image after filtering")
                
            if debug_callback:
                debug_callback({'edges': edges, 'contours': valid_contours})

            lengths = []
            for pts in valid_contours:
                if len(pts) < 2:
                    lengths.append(0)
                else:
                    d = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                    lengths.append(d)
                    
            total_len = sum(lengths)
            if total_len == 0:
                return np.zeros((K, 2))
                
            counts = []
            for l in lengths:
                c = int(np.round(K * l / total_len))
                counts.append(c)
                
            diff = K - sum(counts)
            if diff != 0:
                idx_max = np.argmax(lengths)
                counts[idx_max] += diff
                
            all_samples = []
            for i, pts in enumerate(valid_contours):
                c = counts[i]
                if c <= 0: continue
                    
                if smooth:
                    sampled = smooth_contour_spline(pts, c)
                else:
                    sampled = sample_polyline_uniform(pts, c)
                all_samples.append(sampled)
                
            if not all_samples:
                 return np.zeros((K, 2))
                 
            result = np.vstack(all_samples)
            
            if len(result) > K:
                result = result[:K]
            elif len(result) < K:
                pad = np.tile(result[-1], (K - len(result), 1))
                result = np.vstack([result, pad])
                
            return result
            
        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")
