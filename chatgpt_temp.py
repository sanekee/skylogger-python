import cv2
import numpy as np

def detect_ssd_digit(image):
    """
    Detects the digit displayed on a seven-segment display.
    
    Args:
        image (numpy.ndarray): Input image of the seven-segment display (grayscale).
    
    Returns:
        int: Detected digit (0-9) or -1 if no valid digit is detected.
    """
    def preprocess_image(image):
        """Preprocess the image: thresholding and noise removal."""
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned_binary

    def define_zones(image):
        """Define the zones for each segment of the seven-segment display."""
        height, width = image.shape
        zones = {
            "top": image[:int(0.3 * height), int(0.1 * width):int(0.9 * width)],
            "top-right": image[int(0.1 * height):int(0.5 * height), int(0.7 * width):],
            "bottom-right": image[int(0.5 * height):int(0.9 * height), int(0.7 * width):],
            "bottom": image[int(0.7 * height):, int(0.1 * width):int(0.9 * width)],
            "bottom-left": image[int(0.5 * height):int(0.9 * height), :int(0.3 * width)],
            "top-left": image[int(0.1 * height):int(0.5 * height), :int(0.3 * width)],
            "middle": image[int(0.4 * height):int(0.6 * height), int(0.1 * width):int(0.9 * width)],
        }
        return zones

    def detect_segments(zones):
        """Detect which segments are on or off based on contours."""
        segments = []
        for zone_name, zone in zones.items():
            contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            zone_area = zone.shape[0] * zone.shape[1]
            # Check if any contour's area exceeds a threshold percentage of the zone area
            segment_on = any(cv2.contourArea(contour) / zone_area > 0.2 for contour in contours)
            segments.append(1 if segment_on else 0)
        return segments

    def map_segments_to_digit(segments):
        """Map the detected segment pattern to a digit."""
        segment_to_digit = {
            (1, 1, 1, 1, 1, 1, 0): 0,
            (0, 1, 1, 0, 0, 0, 0): 1,
            (1, 1, 0, 1, 1, 0, 1): 2,
            (1, 1, 1, 1, 0, 0, 1): 3,
            (0, 1, 1, 0, 0, 1, 1): 4,
            (1, 0, 1, 1, 0, 1, 1): 5,
            (1, 0, 1, 1, 1, 1, 1): 6,
            (1, 1, 1, 0, 0, 0, 0): 7,
            (1, 1, 1, 1, 1, 1, 1): 8,
            (1, 1, 1, 1, 0, 1, 1): 9,
        }
        return segment_to_digit.get(tuple(segments), -1)

    # Preprocess the image
    binary_image = preprocess_image(image)
    
    # Define zones
    zones = define_zones(binary_image)
    
    # Detect which segments are on
    segments = detect_segments(zones)
    
    # Map the detected segments to a digit
    digit = map_segments_to_digit(segments)
    
    return digit

# Example usage:
if __name__ == "__main__":
    # Load a grayscale image of a seven-segment display
    input_image = cv2.imread("seven_segment_display.jpg", cv2.IMREAD_GRAYSCALE)
    detected_digit = detect_ssd_digit(input_image)
    print(f"Detected digit: {detected_digit}")
