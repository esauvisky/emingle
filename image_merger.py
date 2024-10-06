import glob
import os
import time

import numpy as np
from PIL import Image
from loguru import logger


class ImageMerger:
    @staticmethod
    def find_image_overlap(base_array, new_array, threshold=0.9):
        logger.debug(f"Entering find_image_overlap with threshold {threshold}")

        height_base, width = base_array.shape
        height_new = new_array.shape[0]

        shifts = np.arange(-height_new + 1, height_base)
        match_percentages = np.zeros(shifts.shape[0], dtype=np.float32)

        for i in range(shifts.shape[0]):
            shift = shifts[i]
            start_base = max(0, shift)
            end_base = min(height_base, shift + height_new)
            start_new = max(0, -shift)
            end_new = start_new + (end_base - start_base)

            if end_base - start_base > 0:
                base_overlap = base_array[start_base:end_base]
                new_overlap = new_array[start_new:end_new]

                # Calculate matching percentage
                match_percentages[i] = np.mean(base_overlap == new_overlap)
                if match_percentages[i] == 1:
                    logger.info(f"Match percentage at shift {shift} is 100%, returning early")
                    return shift, match_percentages[i]

        best_match_index = np.argmax(match_percentages)
        best_match_percentage = match_percentages[best_match_index]
        best_shift = shifts[best_match_index] if best_match_percentage >= threshold else None

        end_time = time.time()
        return best_shift, best_match_percentage

    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.9):
        logger.debug("Entering merge_images_vertically")

        # Convert images to grayscale to reduce data size
        base_img_gray = base_img.convert('L')
        new_img_gray = new_img.convert('L')

        base_array = np.array(base_img)
        new_array = np.array(new_img)
        base_array_gray = np.array(base_img_gray)
        new_array_gray = np.array(new_img_gray)

        if base_array_gray.shape[1] != new_array_gray.shape[1]:
            raise ValueError("Images must have the same width")

        shift, match_percentage = ImageMerger.find_image_overlap(base_array_gray, new_array_gray, threshold)

        if shift is not None:
            logger.success(f"Overlap detected at shift {shift} with match percentage {match_percentage:.2%}")

            height_base = base_array.shape[0]
            height_new = new_array.shape[0]

            if shift < 0:
                merged_array = np.vstack((new_array[0:-shift], base_array))
            else:
                overlap_size = min(height_new, height_base - shift)
                non_overlap_new = new_array[overlap_size:]
                merged_array = np.vstack((base_array, non_overlap_new)) if non_overlap_new.size > 0 else base_array

            return Image.fromarray(merged_array)
        else:
            logger.warning("No overlap detected above the threshold, returning original base image")
            return base_img

    @staticmethod
    def process_single_image(args):
        base_img_path, new_img_path, threshold = args
        base_img = Image.open(base_img_path)
        new_img = Image.open(new_img_path)
        return ImageMerger.merge_images_vertically(base_img, new_img, threshold)




if __name__ == "__main__":
    start_time = time.time()
    logger.info("Image processing started.")

    base_dir = "test"  # Directory containing the images

    # Define patterns to match base and new images
    base_pattern = os.path.join(base_dir, "*_base.png")
    new_pattern = os.path.join(base_dir, "*_new*.png")  # Handles new images with suffixes like _new_1.png

    # Retrieve lists of base and new images
    base_images = glob.glob(base_pattern)
    new_images = glob.glob(new_pattern)

    logger.info(f"Found {len(base_images)} base images and {len(new_images)} new images.")

    # Create a dictionary to map identifiers to base images
    base_dict = {}
    for base in base_images:
        filename = os.path.basename(base)
        identifier = filename.split('_base.png')[0]  # Extract identifier before '_base.png'
        base_dict.setdefault(identifier, []).append(base)

    # Create a dictionary to map identifiers to new images
    new_dict = {}
    for new in new_images:
        filename = os.path.basename(new)
        # Handle new images with possible suffixes like _new_1.png
        identifier = filename.split('_new')[0]
        new_dict.setdefault(identifier, []).append(new)

    total_pairs = 0
    processed_pairs = 0

    # Iterate through each identifier and process corresponding image pairs
    for identifier, base_list in base_dict.items():
        corresponding_new_images = new_dict.get(identifier, [])
        if not corresponding_new_images:
            logger.warning(f"No new images found for base identifier '{identifier}'. Skipping.")
            continue

        for base_path in base_list:
            for new_path in corresponding_new_images:
                total_pairs += 1
                logger.info(f"Processing pair: Base='{base_path}' | New='{new_path}'")
                pair_start_time = time.time()

                # Process the image pair with a threshold of 0.9
                result = ImageMerger.process_single_image((base_path, new_path, 0.9))

                pair_end_time = time.time()
                elapsed = pair_end_time - pair_start_time
                logger.info(f"Processed pair in {elapsed:.2f} seconds.")
                processed_pairs += 1

    end_time = time.time()
    total_elapsed = end_time - start_time
    logger.info(f"Image processing completed. Processed {processed_pairs}/{total_pairs} pairs.")
    logger.info(f"Total execution time: {total_elapsed:.2f} seconds.")
