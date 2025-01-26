import cv2
import numpy as np

def cartoonize_image(img_path, output_path):
    """
    Convert a photo to cartoon style using edge detection and color quantization
    
    Parameters:
    img_path (str): Path to input image
    output_path (str): Path to save output image
    """
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges
    edges = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter for color smoothing while preserving edges
    color = cv2.bilateralFilter(img, 9, 250, 250)
    
    # Combine edge mask with colored image
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Save the output
    cv2.imwrite(output_path, cartoon)
    
    return cartoon

# Example usage
cartoonize_image('input.jpg', 'cartoon_output.jpg')
