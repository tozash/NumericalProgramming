# cp1/src/background_model.py

import numpy as np
import cv2

class BackgroundModelScratch:
    """
    Manual implementation of background subtraction and thresholding.
    """
    def __init__(self, alpha=0.05, threshold=30):
        self.bg = None
        self.alpha = alpha
        self.threshold = threshold
        
    def to_grayscale(self, frame):
        """
        Manual grayscale conversion: 0.299 R + 0.587 G + 0.114 B
        Input: HxWx3 (BGR) -> Output: HxW
        """
        # BGR format
        B = frame[:, :, 0].astype(float)
        G = frame[:, :, 1].astype(float)
        R = frame[:, :, 2].astype(float)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray.astype(np.uint8)
        
    def box_blur(self, image, k=3):
        """
        Manual Box Blur (Moving Average) with kernel size kxk.
        Using integral images or separated filters would be faster,
        but nested loops/sliding windows demonstrate 'from scratch' logic clearly.
        For performance in Python, we'll use a vectorized approach with slicing.
        """
        h, w = image.shape
        pad = k // 2
        padded = np.pad(image, pad, mode='edge').astype(float)
        
        # Vectorized sliding window sum
        # This mimics the convolution operation without using scipy.convolve
        output = np.zeros_like(image, dtype=float)
        
        # Naive implementation is too slow for video in pure Python,
        # so we use standard numpy slicing which is efficient but still 'manual'
        # (i.e., not calling cv2.blur)
        for i in range(k):
            for j in range(k):
                output += padded[i:h+i, j:w+j]
        
        output /= (k*k)
        return output.astype(np.uint8)

    def apply(self, frame):
        gray = self.to_grayscale(frame)
        blurred = self.box_blur(gray, k=5)
        
        if self.bg is None:
            self.bg = blurred.astype(float)
            return np.zeros_like(blurred)
            
        # Update background: bg = alpha * bg + (1-alpha) * current
        self.bg = (1 - self.alpha) * self.bg + self.alpha * blurred.astype(float)
        
        # Absolute difference
        diff = np.abs(self.bg - blurred).astype(np.uint8)
        
        # Thresholding
        # mask = 1 if diff > T else 0
        mask = np.where(diff > self.threshold, 255, 0).astype(np.uint8)
        
        return mask

class BackgroundModelLib:
    """
    Library-based background subtraction using OpenCV.
    """
    def __init__(self, alpha=0.05, threshold=30):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=threshold, detectShadows=False)
        
    def apply(self, frame):
        # MOG2 is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
        mask = self.fgbg.apply(frame)
        
        # Cleanup noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

