import cv2
import numpy as np
from typing import Tuple, Optional

class ImageStyler:
    def __init__(self, img_path: str):
        """Initialize with an image path."""
        self.original = cv2.imread(img_path)
        if self.original is None:
            raise ValueError("Could not load image")
        
    def _adjust_saturation(self, image: np.ndarray, value: float) -> np.ndarray:
        """Adjust the saturation of an image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def cartoon_style(self, edge_size: int = 9, color_levels: int = 7) -> np.ndarray:
        """Create a cartoon effect with enhanced edges and color quantization."""
        # Color quantization
        color = cv2.bilateralFilter(self.original, 9, 250, 250)
        color = color // color_levels * color_levels + color_levels // 2
        
        # Edge detection
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        edges = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 
                                    edge_size, 9)
        
        # Combine
        return cv2.bitwise_and(color, color, mask=edges)

    def anime_style(self, edge_threshold: int = 100) -> np.ndarray:
        """Create an anime-like effect with strong edges and simplified colors."""
        # Color simplification
        color = cv2.bilateralFilter(self.original, 15, 75, 75)
        color = self._adjust_saturation(color, 1.2)  # Increase saturation
        
        # Edge detection
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        
        # Dilate edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create white background
        result = np.full_like(self.original, 255)
        
        # Copy color where there are no edges
        result[edges == 0] = color[edges == 0]
        
        return result

    def watercolor_style(self) -> np.ndarray:
        """Create a watercolor-like effect."""
        # Apply bilateral filter multiple times
        result = self.original
        for _ in range(2):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        # Enhance colors slightly
        result = self._adjust_saturation(result, 1.1)
        
        # Add subtle texture
        noise = np.random.normal(0, 2, result.shape).astype(np.uint8)
        result = cv2.add(result, noise)
        
        return result

    def sketch_style(self, ksize: int = 21) -> np.ndarray:
        """Create a pencil sketch effect."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Invert
        inv = 255 - gray
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(inv, (ksize, ksize), 0)
        
        # Blend using color dodge
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def save_image(self, output_path: str, image: Optional[np.ndarray] = None) -> None:
        """Save the image to specified path."""
        if image is None:
            image = self.original
        cv2.imwrite(output_path, image)

# Example usage:

styler = ImageStyler('input.jpg')

# Generate different styles
cartoon = styler.cartoon_style()
anime = styler.anime_style()
watercolor = styler.watercolor_style()
sketch = styler.sketch_style()

# Save results
styler.save_image('cartoon_output.jpg', cartoon)
styler.save_image('anime_output.jpg', anime)
styler.save_image('watercolor_output.jpg', watercolor)
styler.save_image('sketch_output.jpg', sketch)

