from PIL import Image
import numpy as np

def oil_painting_effect(input_path, output_path, radius=4, intensity=20):
    """
    Apply oil painting effect to an image while preserving transparency
    
    Parameters:
    input_path (str): Path to input PNG image
    output_path (str): Path to save the processed image
    radius (int): Radius of the effect (default: 4)
    intensity (int): Intensity levels for the effect (default: 20)
    """
    # Open image and convert to RGBA if not already
    img = Image.open(input_path).convert('RGBA')
    
    # Split the image into color and alpha channels
    color = np.array(img.convert('RGB'))
    alpha = np.array(img.split()[3])
    
    # Get image dimensions
    height, width = color.shape[:2]
    
    # Create output array
    out = np.zeros_like(color)
    
    # Apply oil painting effect
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            # Get neighborhood
            window = color[i-radius:i+radius+1, j-radius:j+radius+1]
            
            # Calculate intensity bins for each channel
            for c in range(3):
                channel = window[:,:,c].flatten()
                bins = np.zeros(intensity)
                
                # Count intensity frequencies
                for pixel in channel:
                    # 修正：使用 float 來避免溢出，並確保結果在合理範圍內
                    bin_idx = int((float(pixel) * (intensity - 1)) / 255.0)
                    bin_idx = max(0, min(bin_idx, intensity - 1))  # 確保索引在有效範圍內
                    bins[bin_idx] += 1
                
                # Find most frequent intensity
                max_bin = np.argmax(bins)
                out[i,j,c] = int((max_bin * 255.0) / (intensity - 1))
    
    # Convert back to PIL Image
    result = Image.fromarray(out.astype(np.uint8))
    
    # Restore alpha channel
    result.putalpha(Image.fromarray(alpha))
    
    # Save result
    result.save(output_path, 'PNG')
    
    return result

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual image paths
    input_image = "input/input.png"
    output_image = "output.png"
    
    # Apply effect
    oil_painting_effect(input_image, output_image, 1, 5)
