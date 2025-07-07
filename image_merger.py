import filecmp
import glob
import itertools
from math import sqrt
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel
import os
import random
import time
from skimage.feature import match_template

from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from utils import Config


class ImageMerger:
    @staticmethod
    def find_fixed_borders(images, margin=2):
        rgb_arrays = [np.array(img) for img in images]

        top, bottom, left, right = 0, rgb_arrays[0].shape[0], 0, rgb_arrays[0].shape[1]

        for array in rgb_arrays:
            for i in range(top, array.shape[0]):
                if not np.allclose(array[i, :, :], rgb_arrays[0][i, :, :], rtol=margin):
                    top = i
                    break
            for i in range(bottom - 1, -1, -1):
                if not np.allclose(array[i, :, :], rgb_arrays[0][i, :, :], rtol=margin):
                    bottom = i + 1
                    break
            for i in range(left, array.shape[1]):
                if not np.allclose(array[:, i, :], rgb_arrays[0][:, i, :], rtol=margin):
                    left = i
                    break
            for i in range(right - 1, -1, -1):
                if not np.allclose(array[:, i, :], rgb_arrays[0][:, i, :], rtol=margin):
                    right = i + 1
                    break

        return top, bottom, left, right

    @staticmethod
    def remove_borders(array, top, bottom, left, right):
        return array[top:bottom, left:right]

    @staticmethod
    def find_image_overlap(base_array_gray, new_array_gray, threshold=0.8, z_score_threshold=2.0):
        from skimage.metrics import structural_similarity as ssim
        from scipy.ndimage import sobel

        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        best_shift = None
        best_score = 0  # Initialize with a low value (we'll maximize instead of minimize)
        best_zscore = 0

        fig = None
        axs = []
        if Config["DEBUG_MODE"]:
            fig, axs = plt.subplots(1, 3, figsize=(10, 8))  # Create three subplots side by side

        shifts = []
        scores = []
        zscores = []

        # We're not using edge detection anymore since we're only using SSIM

        # Define the range for shifts with a reasonable margin
        margin = 5

        # Prioritize checking shifts where we expect overlaps
        # First check negative shifts (new image above base image)
        # Then check positive shifts (new image below base image)
        shift_ranges = [
            range(-height_new + 1 + margin, 0),  # Negative shifts
            range(height_base - margin, 0, -1)       # Positive shifts
        ]
        shift_range = [val for pair in zip(*shift_ranges) for val in pair]

        for shift in shift_range:
            if shift >= 0:
                # Overlapping regions (new image below base image)
                overlap_height = min(height_base - shift, height_new)
                if overlap_height <= 0:
                    continue
                base_overlap = base_array_gray[shift:shift + overlap_height, :]
                new_overlap = new_array_gray[:overlap_height, :]
            else:
                # Negative shift: new image above base image
                overlap_height = min(height_new + shift, height_base)
                if overlap_height <= 0:
                    continue
                base_overlap = base_array_gray[:overlap_height, :]
                new_overlap = new_array_gray[-shift:-shift + overlap_height, :]

            # Ensure the overlapping regions have the same dimensions
            if base_overlap.shape != new_overlap.shape:
                continue

            # # Skip if overlap is too small
            if overlap_height < 7:
                continue

            # Calculate SSIM for structural similarity (ranges from -1 to 1, higher is better)
            ssim_score = ssim(base_overlap, new_overlap, data_range=255)

            # Use SSIM score directly as our combined score
            combined_score = ssim_score

            shifts.append(shift)
            scores.append(combined_score)

            # Calculate z-score if we have enough data points
            if len(scores) >= 5:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                z_score = (combined_score - mean_score) / (std_score + 1e-8)
                zscores.append(z_score)
            else:
                z_score = 0
                zscores.append(0)

            # Update best score
            if combined_score > best_score:
                best_score = combined_score
                best_shift = shift
                best_zscore = z_score

                # Early termination if we find an exceptionally good match
                if combined_score > 0.95 and z_score > 2.0:
                    logger.info(f"Excellent match found at shift {shift} with score {combined_score:.4f} and z-score {z_score:.2f}")
                    return shift, combined_score, z_score

            # Visualization for debugging
            if Config["DEBUG_MODE"] and (len(shifts) % 43 == 0 or combined_score > 0.9):
                overlap_size_percentage = overlap_height / height_new

                plt.suptitle(f"Shift: {shift}\n"
                            f"Overlap Size: {overlap_size_percentage:.0%}\n"
                            f"Combined Score: {combined_score:.4f}\n"
                            f"Z-score: {z_score:.2f}")

                logger.debug(f"Shift: {shift}. Overlap height: {overlap_height}, "
                            f"Combined Score: {combined_score:.4f}, Z-score: {z_score:.2f}")

                if len(axs) >= 3:
                    axs[0].clear()
                    axs[1].clear()
                    axs[2].clear()

                    # Display the overlapping regions side by side
                    axs[0].imshow(base_overlap, cmap='gray')
                    axs[0].set_title("Base Overlap")
                    axs[0].axis('off')

                    axs[1].imshow(new_overlap, cmap='gray')
                    axs[1].set_title("New Overlap")
                    axs[1].axis('off')

                    # Plot scores against shifts
                    axs[2].set_title("Scores vs. Shift")
                    axs[2].set_xlabel("Shift")
                    axs[2].set_ylabel("Score")
                    axs[2].scatter(shifts, scores, color='blue', alpha=0.5, s=5)
                    axs[2].scatter(best_shift, best_score, color='red', s=50)

                    plt.draw()
                    plt.pause(0.00001)

        if Config["DEBUG_MODE"] and fig is not None:
            plt.close(fig)

        logger.info(f"Best match: shift={best_shift}, score={best_score:.4f}, z-score={best_zscore:.2f}")

        # Return the best shift if it meets the threshold
        if best_score >= threshold and best_zscore > z_score_threshold:
            return best_shift, best_score, best_zscore
        return None, best_score, best_zscore


    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.6):
        # Convert images to grayscale
        base_array_gray = np.array(base_img.convert('L'))
        new_array_gray = np.array(new_img.convert('L'))

        # Ensure images have the same width
        if base_array_gray.shape[1] != new_array_gray.shape[1]:
            raise ValueError(f"Images must have the same width. Base width: {base_array_gray.shape[1]}, New width: {new_array_gray.shape[1]}")

        shift, match_score, zscore = ImageMerger.find_image_overlap(base_array_gray, new_array_gray, threshold)

        base_array = np.array(base_img)
        new_array = np.array(new_img)

        def visualize(image):
            # Visualize the merged image in real-time
            plt.imshow(image)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)  # Adjust pause duration as needed
            plt.clf()  # Clear the figure for the next update

        if shift is not None:
            merged_array_parts = []

            if shift >= 0:
                # Positive shift: new image aligns below the base image
                overlap_height = min(base_array.shape[0] - shift, new_array.shape[0])
                overlap_start_in_base = shift
                overlap_end_in_base = shift + overlap_height

                # base_overlap = base_array[overlap_start_in_base:overlap_end_in_base]
                new_overlap = new_array[:overlap_height]
                blended_overlap = new_overlap

                # blended_overlap = ImageMerger.blend_overlap(base_overlap, new_overlap)

                # Add the part of the base image before the overlapping region
                if shift > 0:
                    merged_array_parts.append(base_array[:shift])

                merged_array_parts.append(blended_overlap)

                # Determine any remaining parts after the overlapping region
                base_remaining = base_array[overlap_end_in_base:]
                new_remaining = new_array[overlap_height:]

                if base_remaining.shape[0] > 0 and new_remaining.shape[0] == 0:
                    # Only base image has remaining data
                    merged_array_parts.append(base_remaining)
                elif base_remaining.shape[0] == 0 and new_remaining.shape[0] > 0:
                    # Only new image has remaining data
                    merged_array_parts.append(new_remaining)
                elif base_remaining.shape[0] > 0 and new_remaining.shape[0] > 0:
                    # Both images have remaining data
                    # Decide which one to include (here we include both)
                    merged_array_parts.append(base_remaining)
                    merged_array_parts.append(new_remaining)
                # If both are empty, do nothing

            else:
                # Negative shift: new image starts before the base image
                overlap_height = min(new_array.shape[0] + shift, base_array.shape[0])
                overlap_start_in_new = -shift
                overlap_end_in_new = overlap_start_in_new + overlap_height

                base_overlap = base_array[:overlap_height]
                new_overlap = new_array[overlap_start_in_new:overlap_end_in_new]

                blended_overlap = ImageMerger.blend_overlap(base_overlap, new_overlap)

                # Add the part of the new image before the overlapping region
                if overlap_start_in_new > 0:
                    merged_array_parts.append(new_array[:overlap_start_in_new])

                merged_array_parts.append(blended_overlap)

                # Determine any remaining parts after the overlapping region
                base_remaining = base_array[overlap_height:]
                new_remaining = new_array[overlap_end_in_new:]

                if base_remaining.shape[0] > 0 and new_remaining.shape[0] == 0:
                    # Only base image has remaining data
                    merged_array_parts.append(base_remaining)
                elif base_remaining.shape[0] == 0 and new_remaining.shape[0] > 0:
                    # Only new image has remaining data
                    merged_array_parts.append(new_remaining)
                elif base_remaining.shape[0] > 0 and new_remaining.shape[0] > 0:
                    # Both images have remaining data
                    merged_array_parts.append(base_remaining)
                    merged_array_parts.append(new_remaining)
                # If both are empty, do nothing

            # Stack all the parts vertically to form the merged image
            merged_array = np.vstack(merged_array_parts)
            logger.info(f"Overlap detected at shift {shift}, overlap height {overlap_height}, match score {match_score:.8f}, zscore {zscore:.2f}. Merging...")

            if Config["DEBUG_MODE"]:
                visualize(Image.fromarray(merged_array))

            return Image.fromarray(merged_array)
        else:
            # If no overlap is detected, concatenate the images
            logger.warning("No overlap detected above the threshold, returning original base image")
            if Config["DEBUG_MODE"]:
                visualize(base_img)
            return base_img

    @staticmethod
    def blend_overlap(base_overlap, new_overlap):
        # Create a linear gradient for blending
        height, width, _ = base_overlap.shape
        alpha = np.linspace(0, 1, height).reshape(height, 1)
        alpha = np.repeat(alpha, width, axis=1)
        alpha = np.expand_dims(alpha, axis=2)  # Make it (height, width, 1) to match RGB channels

        # Blend the images
        blended_overlap = (1 - alpha) * base_overlap + alpha * new_overlap
        return blended_overlap.astype(np.uint8)


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Image processing started.")

    base_dir = "test"  # Directory containing the images

    tests = sorted(os.listdir(base_dir))
    # random.shuffle(tests)
    for subdir in tests[2:]:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            logger.info(f"Processing directory: {subdir_path}")
            dir_start_time = time.time()

            # Get all image files in the directory
            image_files = sorted(glob.glob(os.path.join(subdir_path, "screenshot_*.png")),
                                 key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if not image_files:
                logger.warning(f"No images found in {subdir_path}")
                continue

            # Start with the first image as the base
            image_files_objs = [Image.open(img_path) for img_path in image_files]
            logger.info(f"Images dimensions: {[img.size for img in image_files_objs]}")
            top, bottom, left, right = ImageMerger.find_fixed_borders(image_files_objs)
            image_files_objs = [
                ImageMerger.remove_borders(np.array(img), top, bottom, left, right) for img in image_files_objs]
            image_files_objs = [Image.fromarray(img) for img in image_files_objs]

            logger.info(f"Images dimensions: {[img.size for img in image_files_objs]}")
            base_img = image_files_objs[0]
            for new_img in image_files_objs[1:]:
                base_img = ImageMerger.merge_images_vertically(base_img, new_img, threshold=0.1)

            # Save the merged image temporarily
            merged_temp_path = os.path.join(subdir_path, "merged_temp.png")
            base_img.save(merged_temp_path)

            # Compare with the existing merged image
            existing_merged_path = os.path.join(subdir_path, "merged_screenshot.png")
            if os.path.exists(existing_merged_path):
                if filecmp.cmp(merged_temp_path, existing_merged_path, shallow=False):
                    logger.success(f"Merged image in {subdir_path} matches the existing merged image.")
                else:
                    logger.error(f"Merged image in {subdir_path} does not match the existing merged image. "
                                 f"Debug {merged_temp_path} and press Return to continue.")
                    input()
            else:
                logger.warning(f"Existing merged image not found at {existing_merged_path}. Saving merged image.")

            # Clean up the temporary merged image
            if os.path.exists(merged_temp_path):
                os.remove(merged_temp_path)

            logger.info(f"Completed processing directory: {subdir_path} in {time.time() - dir_start_time:.2f} seconds")

    logger.info("Image processing completed.")
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
