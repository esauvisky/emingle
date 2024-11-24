import filecmp
import glob
import math
import os
import time

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from loguru import logger


class ImageMerger:
    @staticmethod
    def find_image_overlap(base_array, new_array, threshold=0.05):
        height_base, width = base_array.shape
        height_new = new_array.shape[0]

        shifts = np.arange(-height_new + 1, height_base + height_new)
        # Retain only the first 'height_new' * 2 and last 'height_new' * 2 shifts
        # as anything beyond that is a redundant overlap
        shifts = np.concatenate([shifts[:height_new * 2], shifts[-height_new * 2:]])
        match_percentages = np.zeros(shifts.shape[0], dtype=np.float32)

        for i in range(shifts.shape[0]):
            shift = shifts[i]
            overlap_size = min(height_new, height_base - shift)
            overlap_size_percentage = overlap_size / height_new
            if overlap_size_percentage == 1:
                continue

            start_base = max(0, shift)
            end_base = min(height_base, shift + height_new)
            start_new = max(0, -shift)
            end_new = start_new + (end_base - start_base)

            if end_base - start_base > 0:
                base_overlap = base_array[start_base:end_base]
                new_overlap = new_array[start_new:end_new]

                # Calculate matching percentage
                match_percentages[i] = np.mean(base_overlap == new_overlap)
                logger.debug(f"Start Base: {start_base}\tEnd Base: {end_base}\tStart New: {start_new}\tEnd New: {end_new}\tShift: {shift}\tOverlap Size: {overlap_size}\tOverlap Size Percentage: {overlap_size_percentage}\tMatch Percentage: {match_percentages[i]:.2%}")
                # mse = np.mean(((new_overlap - base_overlap)) ** 2)
                # original_match_percentage = 1 - (mse / 255**2)  # Normalize to [0,1]
                # match_percentages[i] = original_match_percentage

        best_match_index = np.argmax(match_percentages)
        best_match_percentage = match_percentages[best_match_index]
        best_shift = shifts[best_match_index] if best_match_percentage >= threshold else None

        end_time = time.time()
        return best_shift, best_match_percentage

    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.1):
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
            height_base = base_array.shape[0]
            height_new = new_array.shape[0]
            overlap_size = min(height_new, height_base - shift)

            if overlap_size < math.ceil(height_new * 0.1):
                logger.warning(f"Overlap detected at height {shift} with {overlap_size} rows of overlap and match percentage {match_percentage:.2%}. Too small to merge.")
                return base_img
            else:
                logger.success(f"Overlap detected at height {shift} with {overlap_size} rows of overlap and match percentage {match_percentage:.2%}. Merging...")

            if shift < 0:
                merged_array = np.vstack((new_array[0:-shift], base_array))
            else:
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

    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            logger.info(f"Processing directory: {subdir_path}")
            dir_start_time = time.time()

            # Get all image files in the directory
            image_files = sorted(glob.glob(os.path.join(subdir_path, "screenshot_*.png")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if not image_files:
                logger.warning(f"No images found in {subdir_path}")
                continue

            # Start with the first image as the base
            base_img_path = image_files[0]
            base_img = Image.open(base_img_path)
            image_files_objs = [Image.open(img_path) for img_path in image_files]
            # identical_pixels = ImageMerger.find_identical_pixels(image_files_objs)

            # Merge all subsequent images with real-time visualization
            for new_img in image_files_objs[1:]:
                base_img = ImageMerger.merge_images_vertically(
                    base_img, new_img
                )

                # Visualize the merged image in real-time
                plt.imshow(base_img)
                plt.title(f"Merging in {subdir_path}")
                plt.axis('off')
                plt.draw()
                plt.pause(0.01)  # Adjust pause duration as needed
                plt.clf()  # Clear the figure for the next update

            # Save the merged image temporarily
            merged_temp_path = os.path.join(subdir_path, "merged_temp.png")
            base_img.save(merged_temp_path)

            # Compare with the existing merged image
            existing_merged_path = os.path.join(subdir_path, "merged_screenshot.png")
            if os.path.exists(existing_merged_path):
                if filecmp.cmp(merged_temp_path, existing_merged_path, shallow=False):
                    logger.success(
                        f"Merged image in {subdir_path} matches the existing merged image."
                    )
                else:
                    logger.error(
                        f"Merged image in {subdir_path} does not match the existing merged image. "
                        f"Debug {merged_temp_path} and press Return to continue."
                    )
                    input()
            else:
                logger.warning(
                    f"Existing merged image not found at {existing_merged_path}. Saving merged image."
                )

            # Clean up the temporary merged image
            if os.path.exists(merged_temp_path):
                os.remove(merged_temp_path)

    logger.info("Image processing completed.")
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
