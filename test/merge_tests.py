import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_overlap(base_array, new_array, threshold=0.95):
    """
    Detect if the new image overlaps at the top or bottom of the base image.

    Args:
    base_array (np.array): The base image array
    new_array (np.array): The new image array to be merged
    threshold (float): The similarity threshold for considering a match (0-1)

    Returns:
    tuple: (overlap_position, overlap_size) where overlap_position is 'top', 'bottom', or None,
           and overlap_size is the number of overlapping rows or None if no overlap
    """
    try:
        if base_array.shape[1] != new_array.shape[1]:
            raise ValueError("Images must have the same width")

        # Check for overlap at the top
        for i in range(10, min(base_array.shape[0], new_array.shape[0]) + 1):
            if np.mean(base_array[:i] == new_array[-i:]) >= threshold:
                return 'top', i

        # Check for overlap at the bottom
        for i in range(10, min(base_array.shape[0], new_array.shape[0]) + 1):
            if np.mean(base_array[-i:] == new_array[:i]) >= threshold:
                return 'bottom', i

        return None, None
    except Exception as e:
        logger.error(f"Error in detect_overlap: {str(e)}")
        raise

def merge_images_vertically(base_img, new_img):
    """
    Merge two images vertically if they overlap at the top or bottom, extending the base image.
    If no overlap, return the original base image.

    Args:
    base_img (PIL.Image): The base image
    new_img (PIL.Image): The new image to be merged

    Returns:
    PIL.Image: The merged image or the original base image if no overlap
    """
    try:
        # Convert images to numpy arrays
        base_array = np.array(base_img)
        new_array = np.array(new_img)

        # Check if images are in the same mode (e.g., RGB, RGBA)
        if base_img.mode != new_img.mode:
            raise ValueError(f"Image modes don't match: {base_img.mode} vs {new_img.mode}")

        # Ensure images have the same width
        if base_array.shape[1] != new_array.shape[1]:
            raise ValueError("Images must have the same width")

        # Detect overlap
        overlap_position, overlap_size = detect_overlap(base_array, new_array)

        if overlap_position == 'top':
            logger.info(f"Overlap detected at the top, {overlap_size} rows at {overlap_position}")
            merged_array = np.vstack((new_array[:-overlap_size], base_array))
            return Image.fromarray(merged_array)
        elif overlap_position == 'bottom':
            logger.info(f"Overlap detected at the bottom, {overlap_size} rows at {overlap_position}")
            merged_array = np.vstack((base_array, new_array[overlap_size:]))
            return Image.fromarray(merged_array)
        else:
            logger.info("No overlap detected, returning original base image")
            return base_img

    except Exception as e:
        logger.error(f"Error in merge_images_vertically: {str(e)}")
        raise

# Usage example
if __name__ == "__main__":
    try:
        base_img = Image.open("test/2_base.png")
        new_img = Image.open("test/2_new.png")

        result = merge_images_vertically(base_img, new_img)
        result.show()
        logger.info("Images merged successfully")
    except Exception as e:
        logger.error(f"Failed to merge images: {str(e)}")
