import cv2
import numpy as np
import os

def create_lemon():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Draw a yellow ellipse
    cv2.ellipse(img, (300, 200), (150, 100), 0, 0, 360, (0, 255, 255), -1)
    
    # Save
    os.makedirs('assets', exist_ok=True)
    cv2.imwrite('assets/input_lemon.jpg', img)
    print("Created assets/input_lemon.jpg")

if __name__ == "__main__":
    create_lemon()
