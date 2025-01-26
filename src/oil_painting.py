import cv2
import numpy as np
from typing import Tuple

def oil_painting_effect(
    image_path: str,
    output_path: str,
    radius: int = 4,
    intensity_levels: int = 20
) -> None:
    """
    Apply oil painting effect to an image.
    
    Parameters:
    - image_path: Path to input image
    - output_path: Path to save the result
    - radius: Radius of the pixel neighborhood
    - intensity_levels: Number of intensity levels
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # Convert to float32
    img_float = img.astype(np.float32) / 255.0
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create output image
    result = np.zeros_like(img_float)
    
    # Process each pixel
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            # Get neighborhood
            neighborhood = img_float[y-radius:y+radius+1, x-radius:x+radius+1]
            
            # Calculate intensity for each channel
            for c in range(3):  # BGR channels
                # Get current channel
                channel = neighborhood[:, :, c]
                
                # Calculate intensity levels
                intensities = (channel * intensity_levels).astype(int)
                
                # Find most frequent intensity
                unique_intensities, counts = np.unique(intensities, return_counts=True)
                dominant_intensity = unique_intensities[np.argmax(counts)]
                
                # Set the result
                result[y, x, c] = dominant_intensity / intensity_levels
    
    # Convert back to uint8
    result = (result * 255).astype(np.uint8)
    
    # Apply bilateral filter to smooth while preserving edges
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    # Enhance colors slightly
    result = cv2.convertScaleAbs(result, alpha=1.2, beta=0)
    
    # Save result
    cv2.imwrite(output_path, result)

def create_artistic_oil_painting(
    image_path: str,
    output_path: str,
    style_intensity: str = 'medium'
) -> None:
    """
    Create oil painting effect with predefined style intensities.
    
    Parameters:
    - image_path: Path to input image
    - output_path: Path to save the result
    - style_intensity: 'light', 'medium', or 'heavy'
    """
    # Style parameters
    style_params = {
        'light': {'radius': 2, 'intensity_levels': 10},
        'medium': {'radius': 4, 'intensity_levels': 20},
        'heavy': {'radius': 6, 'intensity_levels': 30}
    }
    
    if style_intensity not in style_params:
        raise ValueError("Style intensity must be 'light', 'medium', or 'heavy'")
    
    params = style_params[style_intensity]
    oil_painting_effect(
        image_path,
        output_path,
        radius=params['radius'],
        intensity_levels=params['intensity_levels']
    )

# Example usage:

# Basic usage
oil_painting_effect('input/Chun_rembg.png', 'Chun_rembg_oil.png', 1, 5)

# Or use the artistic version with different intensities
# create_artistic_oil_painting('input/Chun.jpg', 'output_light.jpg', 'light')
# create_artistic_oil_painting('input/Chun.jpg', 'output_medium.jpg', 'medium')
# create_artistic_oil_painting('input/Chun.jpg', 'output_heavy.jpg', 'heavy')

