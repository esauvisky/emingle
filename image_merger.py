from loguru import logger
import numpy as np
from PIL import Image

class ImageMerger:
    @staticmethod
    def detect_overlap(base_array, new_array, threshold=1):
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

            base_height = base_array.shape[0]
            new_height = new_array.shape[0]

            best_overlap = {'position': None, 'size': 0, 'mean': 0}

            # Check for overlap at the top, starting with the new image
            # placed at the top of the base image and moving upwards
            for i in range(new_height, 10, -1):
                mean = np.mean(base_array[:i] == new_array[-i:])
                if mean == 1:
                    return 'top', i
                if mean > threshold and mean > best_overlap['mean']:
                    best_overlap = {'position': 'top', 'size': i, 'mean': mean}

            # Check for overlap at the bottom, starting with the new image
            # placed at the bottom of the base image and moving downwards
            for i in range(10, new_height):
                mean = np.mean(base_array[-i:] == new_array[:i])
                if mean == 1:
                    return 'bottom', i
                if mean > threshold and mean > best_overlap['mean']:
                    best_overlap = {'position': 'bottom', 'size': i, 'mean': mean}

            if best_overlap['position']:
                return best_overlap['position'], best_overlap['size']
            return None, None
        except Exception as e:
            logger.error(f"Error in detect_overlap: {str(e)}")
            raise

    @staticmethod
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
            overlap_position, overlap_size = ImageMerger.detect_overlap(base_array, new_array)

            if overlap_position == 'top':
                logger.info(f"Overlap detected at the top, {overlap_size} rows at {overlap_position}")
                merged_array = np.vstack((new_array[:-overlap_size], base_array)) # type: ignore
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

    @staticmethod
    def process_images(base_img_path, new_img_path):
        try:
            base_img = Image.open(base_img_path)
            new_img = Image.open(new_img_path)

            result = ImageMerger.merge_images_vertically(base_img, new_img)
            result.show()
            logger.info("Images merged successfully")
        except Exception as e:
            logger.error(f"Failed to merge images: {str(e)}")

# Usage example
if __name__ == "__main__":
    ImageMerger.process_images("test/2_base.png", "test/2_new.png")
