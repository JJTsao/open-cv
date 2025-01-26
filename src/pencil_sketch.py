import cv2
import numpy as np
from typing import Tuple, Optional

class PencilSketchGenerator:
    def __init__(self, image_path: str):
        """Initialize with image path."""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")

    def create_sketch(self, 
                     style: str = 'classic',
                     intensity: float = 0.5,
                     blur_size: int = 5) -> np.ndarray:
        """
        Create pencil sketch effect.
        
        Parameters:
        - style: 'classic', 'detailed', or 'artistic'
        - intensity: Strength of the effect (0.0 to 1.0)
        - blur_size: Size of the blur kernel (odd number)
        """
        # Ensure blur_size is odd
        blur_size = max(3, blur_size + (blur_size % 2 == 0))
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        if style == 'classic':
            # Invert grayscale image
            inverted = 255 - gray
            
            # Create blurred version
            blurred = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)
            
            # Blend using color dodge
            sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
            
        elif style == 'detailed':
            # Apply bilateral filter for edge preservation
            smooth = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Create edge mask using Laplacian
            edges = cv2.Laplacian(smooth, cv2.CV_8U, ksize=5)
            
            # Enhance edges
            edges = cv2.convertScaleAbs(edges, alpha=1.5, beta=0)
            
            # Invert and blend
            sketch = 255 - edges
            
        elif style == 'artistic':
            # Create two differently blurred versions
            blur1 = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            blur2 = cv2.GaussianBlur(gray, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
            
            # Blend the two blurred versions
            sketch = cv2.addWeighted(blur1, 1.5, blur2, -0.5, 0)
            
            # Add some texture
            noise = np.random.normal(0, 25, sketch.shape).astype(np.uint8)
            sketch = cv2.add(sketch, noise)
            
        else:
            raise ValueError("Unknown style. Use 'classic', 'detailed', or 'artistic'")
        
        # Apply intensity adjustment
        sketch = cv2.convertScaleAbs(sketch, alpha=intensity, beta=255 * (1 - intensity))
        
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def create_colored_sketch(self, 
                            color_strength: float = 0.3,
                            blur_size: int = 5) -> np.ndarray:
        """
        Create colored pencil sketch effect.
        
        Parameters:
        - color_strength: Strength of original colors (0.0 to 1.0)
        - blur_size: Size of the blur kernel
        """
        # Get grayscale sketch
        gray_sketch = self.create_sketch(style='classic', 
                                       intensity=0.7, 
                                       blur_size=blur_size)
        
        # Convert to grayscale
        gray_sketch = cv2.cvtColor(gray_sketch, cv2.COLOR_BGR2GRAY)
        
        # Create color layer
        color = cv2.bilateralFilter(self.original, 9, 250, 250)
        
        # Blend original color with sketch
        colored_sketch = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(colored_sketch, 1 - color_strength, 
                                color, color_strength, 0)
        
        return result

    def create_hatched_sketch(self, 
                            line_size: int = 8,
                            angle: float = 45.0) -> np.ndarray:
        """
        Create hatched pencil sketch effect.
        
        Parameters:
        - line_size: Size of hatching lines
        - angle: Angle of hatching lines in degrees
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Create hatching pattern
        pattern = np.zeros((line_size * 2, line_size * 2), dtype=np.uint8)
        center = line_size
        cv2.line(pattern, (0, center), (line_size * 2, center), 255, 1)
        
        # Rotate pattern
        M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        pattern = cv2.warpAffine(pattern, M, pattern.shape)
        
        # Create different densities of pattern
        result = np.zeros_like(gray)
        for i in range(4):
            mask = cv2.inRange(gray, i * 64, (i + 1) * 64)
            pattern_scaled = cv2.resize(pattern, None, 
                                      fx=1 - i * 0.2, 
                                      fy=1 - i * 0.2)
            temp = np.zeros_like(gray)
            for y in range(0, gray.shape[0], pattern_scaled.shape[0]):
                for x in range(0, gray.shape[1], pattern_scaled.shape[1]):
                    if y + pattern_scaled.shape[0] <= gray.shape[0] and \
                       x + pattern_scaled.shape[1] <= gray.shape[1]:
                        temp[y:y+pattern_scaled.shape[0], 
                             x:x+pattern_scaled.shape[1]] = pattern_scaled
            result = cv2.add(result, cv2.bitwise_and(temp, temp, mask=mask))
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def save_image(self, output_path: str, image: Optional[np.ndarray] = None) -> None:
        """Save the image to specified path."""
        if image is None:
            image = self.original
        cv2.imwrite(output_path, image)

# Example usage:
generator = PencilSketchGenerator('input/input.jpg')

# Classic sketch
sketch = generator.create_sketch(style='classic')
generator.save_image('classic_sketch.jpg', sketch)

# Detailed sketch
detailed = generator.create_sketch(style='detailed')
generator.save_image('detailed_sketch.jpg', detailed)

# Artistic sketch
artistic = generator.create_sketch(style='artistic')
generator.save_image('artistic_sketch.jpg', artistic)

# Colored sketch
colored = generator.create_colored_sketch(color_strength=0.3)
generator.save_image('colored_sketch.jpg', colored)

# Hatched sketch
hatched = generator.create_hatched_sketch(line_size=8, angle=45.0)
generator.save_image('hatched_sketch.jpg', hatched)
