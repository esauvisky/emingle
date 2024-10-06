from loguru import logger
import numpy as np
from PIL import Image

class ImageMerger:
    @staticmethod
    def find_image_overlap(base_array, new_array, threshold=0.995):
        """
        Find the vertical shift where new_array overlaps base_array with the highest percentage of matching pixels above a threshold.

        Args:
            base_array (np.array): Base image array (height_base, width, channels)
            new_array (np.array): New image array (height_new, width, channels)
            threshold (float): Minimum percentage of matching pixels required (between 0 and 1)

        Returns:
            tuple: (best_shift, best_match_percentage) where best_shift is the vertical shift with the highest match percentage above threshold,
                   or (None, 0) if no such shift is found.
        """
        height_base = base_array.shape[0]
        height_new = new_array.shape[0]
        best_shift = None
        best_match_percentage = 0

        # Shifts range from -(height_new - 1) to height_base - 1
        for shift in range(-height_new + 1, height_base):
            # Calculate overlapping indices in base and new arrays
            start_base = max(0, shift)
            end_base = min(height_base, shift + height_new)

            start_new = max(0, -shift)
            end_new = start_new + (end_base - start_base)

            if end_base - start_base > 0:
                # There is overlap
                base_overlap = base_array[start_base:end_base]
                new_overlap = new_array[start_new:end_new]

                # Calculate the percentage of matching pixels
                if base_array.ndim == 2:
                    # Grayscale image
                    matching_pixels = np.sum(base_overlap == new_overlap)
                    total_pixels = base_overlap.size
                elif base_array.ndim == 3:
                    # Color image
                    matching_pixels = np.sum(np.all(base_overlap == new_overlap, axis=2))
                    total_pixels = base_overlap.shape[0] * base_overlap.shape[1]
                else:
                    raise ValueError("Unsupported image array dimensions.")

                match_percentage = matching_pixels / total_pixels

                if match_percentage > best_match_percentage and match_percentage >= threshold:
                    best_match_percentage = match_percentage
                    best_shift = shift

        return best_shift, best_match_percentage

    @staticmethod
    def detect_overlap(base_array, new_array, threshold=0.995):
        """
        Detect if the new image overlaps the base image with matching overlapping rows above a threshold.

        Args:
            base_array (np.array): The base image array
            new_array (np.array): The new image array to be merged
            threshold (float): Minimum percentage of matching pixels required (between 0 and 1)

        Returns:
            tuple: (best_shift, best_match_percentage) where best_shift is the vertical shift with the highest match percentage above threshold,
                   or (None, 0) if no such shift is found.
        """
        try:
            if base_array.shape[1] != new_array.shape[1]:
                raise ValueError("Images must have the same width")

            shift, match_percentage = ImageMerger.find_image_overlap(base_array, new_array, threshold)

            return shift, match_percentage

        except Exception as e:
            logger.error(f"Error in detect_overlap: {str(e)}")
            raise

    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.995):
        """
        Merge two images vertically if they overlap, extending the base image.
        If no overlap, return the original base image.

        Args:
            base_img (PIL.Image): The base image
            new_img (PIL.Image): The new image to be merged
            threshold (float): Minimum percentage of matching pixels required (between 0 and 1)

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
            shift, match_percentage = ImageMerger.detect_overlap(base_array, new_array, threshold)

            if shift is not None:
                logger.info(f"Overlap detected at shift {shift} with match percentage {match_percentage:.2%}")

                height_base = base_array.shape[0]
                height_new = new_array.shape[0]

                if shift < 0:
                    # New image overlaps the base image starting above it
                    # Non-overlapping part of new_array is from 0 to -shift
                    non_overlap_new = new_array[0 : -shift]
                    merged_array = np.vstack((non_overlap_new, base_array))
                else:
                    # New image overlaps the base image starting at row 'shift'
                    overlap_size = min(height_new, height_base - shift)
                    non_overlap_new = new_array[overlap_size:]

                    if non_overlap_new.size > 0:
                        merged_array = np.vstack((base_array, non_overlap_new))
                    else:
                        merged_array = base_array
                return Image.fromarray(merged_array)
            else:
                logger.info("No overlap detected above the threshold, returning original base image")
                return base_img

        except Exception as e:
            logger.error(f"Error in merge_images_vertically: {str(e)}")
            raise

    @staticmethod
    def process_images(base_img_path, new_img_path, threshold=0.995):
        try:
            base_img = Image.open(base_img_path)
            new_img = Image.open(new_img_path)

            result = ImageMerger.merge_images_vertically(base_img, new_img, threshold)
            result.show()
            logger.info("Images merged successfully")
        except Exception as e:
            logger.error(f"Failed to merge images: {str(e)}")

    # Include the find_image_overlap method in the class
    find_image_overlap = find_image_overlap

# Usage example
if __name__ == "__main__":
    # Adjust the threshold as needed (e.g., 99.5% matching pixels)
    ImageMerger.process_images("test/2_base.png", "test/2_new.png", threshold=0.95)
