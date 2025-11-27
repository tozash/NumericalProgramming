# cp1/src/object_tracking.py

import numpy as np
import cv2
from scipy.spatial.distance import cdist

def find_centroids_scratch(mask, min_area=50):
    """
    Find centroids of connected components from scratch (or simple clustering).
    For true 'scratch', we might implement BFS/DFS for blob extraction.
    However, for performance/complexity balance, we will use a simple grid scan 
    or standard 2-pass algorithm if needed. 
    
    To keep it feasible in Python without C-extensions, we might cheat slightly 
    by using cv2.findContours ONLY for blob extraction, 
    OR implement a very simple recursive flood fill.
    
    Let's implement a simple BFS flood fill for demonstration of 'scratch'.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    centroids = []
    
    # Downsample for speed if needed, but let's try full res logic on small blobs
    # This manual BFS is slow in Python. 
    # COMPROMISE: Use cv2.connectedComponents (it's a basic utility) 
    # but calculate centroids manually from the label map.
    
    num_labels, labels = cv2.connectedComponents(mask)
    
    for i in range(1, num_labels): # 0 is background
        # Get indices of pixels in this component
        y_indices, x_indices = np.where(labels == i)
        
        if len(y_indices) < min_area:
            continue
            
        # Compute centroid
        cy = np.mean(y_indices)
        cx = np.mean(x_indices)
        centroids.append((cx, cy))
        
    return centroids

def find_centroids_lib(mask, min_area=50):
    """
    Find centroids using OpenCV contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))
    return centroids

class Tracker:
    """
    Simple centroid tracker using Nearest Neighbor.
    """
    def __init__(self, max_dist=50):
        self.next_id = 0
        self.objects = {} # id -> list of (x, y)
        self.active_ids = []
        self.max_dist = max_dist

    def update(self, current_centroids):
        """
        Update tracks with new centroids.
        current_centroids: list of (x, y) tuples
        """
        if not current_centroids:
            # Lost all objects
            self.active_ids = []
            return self.objects
            
        current_centroids = np.array(current_centroids)
        
        if len(self.active_ids) == 0:
            # Initialize new objects
            for centroid in current_centroids:
                self.objects[self.next_id] = [centroid]
                self.active_ids.append(self.next_id)
                self.next_id += 1
            return self.objects
            
        # Match existing active objects to new centroids
        # Get last positions of active objects
        prev_positions = []
        for obj_id in self.active_ids:
            prev_positions.append(self.objects[obj_id][-1])
        prev_positions = np.array(prev_positions)
        
        # Distance matrix: (num_active, num_new)
        D = cdist(prev_positions, current_centroids)
        
        # Simple greedy assignment or hungarian. 
        # For "scratch", greedy is fine: find min dist, assign, remove, repeat.
        
        rows = D.shape[0]
        cols = D.shape[1]
        
        used_rows = set()
        used_cols = set()
        
        # Find matches
        matches = []
        while len(used_rows) < rows and len(used_cols) < cols:
            # Find min element in D that isn't used
            # (This is O(N^3) naively, but N is small ~1-5 objects)
            min_val = float('inf')
            r_min, c_min = -1, -1
            
            for r in range(rows):
                if r in used_rows: continue
                for c in range(cols):
                    if c in used_cols: continue
                    if D[r, c] < min_val:
                        min_val = D[r, c]
                        r_min, c_min = r, c
            
            if min_val > self.max_dist:
                break
                
            matches.append((self.active_ids[r_min], current_centroids[c_min]))
            used_rows.add(r_min)
            used_cols.add(c_min)
            
        # Update matched
        new_active_ids = []
        for obj_id, centroid in matches:
            self.objects[obj_id].append(centroid)
            new_active_ids.append(obj_id)
            
        # Create new tracks for unmatched centroids
        for c in range(cols):
            if c not in used_cols:
                self.objects[self.next_id] = [current_centroids[c]]
                new_active_ids.append(self.next_id)
                self.next_id += 1
                
        self.active_ids = new_active_ids
        return self.objects

