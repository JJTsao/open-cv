import cv2
import numpy as np
from typing import Tuple, Optional

class PixelArtConverter:
    def __init__(self, image_path: str):
        """
        Initialize the converter with an image path.
        """
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")
        
    def _reduce_colors(self, image: np.ndarray, num_colors: int) -> np.ndarray:
        """
        Reduce the number of colors in the image using k-means clustering.
        """
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Define criteria for k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Apply k-means clustering
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, 
                                      cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        # Map each pixel to its corresponding center
        quantized = centers[labels.flatten()]
        
        # Reshape back to original image shape
        return quantized.reshape(image.shape)
    
    def create_pixel_art(self, 
                        pixel_size: int = 8, 
                        num_colors: int = 8,
                        edge_enhancement: bool = True) -> np.ndarray:
        """
        Convert image to pixel art style.
        
        Parameters:
        - pixel_size: Size of each pixel block
        - num_colors: Number of colors to use in the final image
        - edge_enhancement: Whether to enhance edges in the final result
        """
        # Get image dimensions
        height, width = self.original.shape[:2]
        
        # Calculate new dimensions
        new_height = height // pixel_size
        new_width = width // pixel_size
        
        # Resize image down
        small = cv2.resize(self.original, (new_width, new_height), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Reduce colors
        if num_colors > 0:
            small = self._reduce_colors(small, num_colors)
        
        # Resize back up with nearest neighbor interpolation
        pixelated = cv2.resize(small, (width, height), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Optional edge enhancement
        if edge_enhancement:
            # Create edge mask
            gray = cv2.cvtColor(pixelated, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
            edges = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate edges slightly
            kernel = np.ones((2,2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Darken pixels along edges
            pixelated[edges > 0] = pixelated[edges > 0] * 0.8
        
        return pixelated
    
    def create_gameboy_style(self, pixel_size: int = 6) -> np.ndarray:
        """
        Create a Game Boy style image with 4 shades of green.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Resize down and up for pixelation
        height, width = gray.shape
        small = cv2.resize(gray, (width // pixel_size, height // pixel_size),
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        
        # Create Game Boy green colors
        gb_colors = np.array([
            [15, 56, 15],    # Darkest
            [48, 98, 48],    # Dark
            [139, 172, 15],  # Light
            [155, 188, 15]   # Lightest
        ], dtype=np.uint8)
        
        # Map grayscale values to 4 levels
        levels = np.zeros_like(pixelated)
        levels[pixelated > 64] = 1
        levels[pixelated > 128] = 2
        levels[pixelated > 192] = 3
        
        # Create RGB image
        result = np.zeros((height, width, 3), dtype=np.uint8)
        for i, color in enumerate(gb_colors):
            mask = levels == i
            result[mask] = color
        
        return result
    
    def create_retro_style(self, 
                          pixel_size: int = 6, 
                          palette: str = 'nes') -> np.ndarray:
        """
        Create retro-style pixel art with predefined color palettes.
        """
        # Predefined color palettes (RGB values)
        palettes = {
            'nes': [
                [0, 0, 0],      # Black
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 128, 0],  # Orange
                [255, 0, 255],  # Magenta
                [255, 255, 255] # White
            ]
        }
        
        if palette not in palettes:
            raise ValueError(f"Unknown palette: {palette}")
        
        # Create pixelated image
        pixelated = self.create_pixel_art(pixel_size, 
                                        num_colors=len(palettes[palette]),
                                        edge_enhancement=False)
        
        # Map colors to palette
        pixels = pixelated.reshape(-1, 3)
        palette_colors = np.array(palettes[palette], dtype=np.uint8)
        
        # Find nearest palette color for each pixel
        distances = np.sqrt(((pixels[:, np.newaxis] - palette_colors) ** 2).sum(axis=2))
        nearest_palette_indices = distances.argmin(axis=1)
        mapped_pixels = palette_colors[nearest_palette_indices]
        
        return mapped_pixels.reshape(pixelated.shape)
    
    def save_image(self, output_path: str, image: Optional[np.ndarray] = None) -> None:
        """Save the image to specified path."""
        if image is None:
            image = self.original
        cv2.imwrite(output_path, image)

# Example usage:
converter = PixelArtConverter('input/input.jpg')

# Basic pixel art
pixel_art = converter.create_pixel_art(pixel_size=3, num_colors=12)
converter.save_image('pixel_art.jpg', pixel_art)

# Game Boy style
gameboy = converter.create_gameboy_style(pixel_size=3)
converter.save_image('gameboy.jpg', gameboy)

# Retro NES style
retro = converter.create_retro_style(pixel_size=3, palette='nes')
converter.save_image('retro.jpg', retro)
