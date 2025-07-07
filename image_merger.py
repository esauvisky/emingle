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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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
    def _process_shift_chunk(base_array_gray, new_array_gray, shift_chunk, threshold, z_score_threshold):
        """Process a chunk of shifts in a separate process"""
        from skimage.metrics import structural_similarity as ssim

        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        best_shift = None
        best_score = 0
        best_zscore = 0

        shifts = []
        scores = []
        zscores = []

        for shift in shift_chunk:
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

            # Skip if overlap is too small
            if overlap_height < 7:
                continue

            # Calculate SSIM for structural similarity
            ssim_score = ssim(base_overlap, new_overlap, data_range=255)
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
                    return shift, combined_score, z_score, True  # True indicates early termination

        # Return the best shift if it meets the threshold
        if best_score >= threshold and best_zscore > z_score_threshold:
            return best_shift, best_score, best_zscore, False  # False indicates normal completion
        return None, best_score, best_zscore, False

    @staticmethod
    def find_image_overlap(base_array_gray, new_array_gray, threshold=0.8, z_score_threshold=2.0):
        from skimage.metrics import structural_similarity as ssim
        from scipy.ndimage import sobel

        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        fig = None
        axs = []
        if Config["DEBUG_MODE"]:
            fig, axs = plt.subplots(1, 3, figsize=(10, 8))  # Create three subplots side by side

        # Define the range for shifts with a reasonable margin
        margin = 5

        # Prioritize checking shifts where we expect overlaps
        # First check negative shifts (new image above base image)
        # Then check positive shifts (new image below base image)
        shift_ranges = [
            range(-height_new + 1 + margin, 0),  # Negative shifts
            range(height_base - margin, 0, -1)   # Positive shifts
        ]
        shift_range = [val for pair in zip(*shift_ranges) for val in pair]

        # Determine number of processes to use (half of CPU cores)
        num_processes = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Using {num_processes} processes for parallel processing")

        # Split the shift_range into chunks for parallel processing
        chunk_size = max(1, len(shift_range) // num_processes)
        shift_chunks = [shift_range[i:i + chunk_size] for i in range(0, len(shift_range), chunk_size)]

        best_result = None

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks
            futures = [executor.submit(
                ImageMerger._process_shift_chunk,
                base_array_gray,
                new_array_gray,
                chunk,
                threshold,
                z_score_threshold
            ) for chunk in shift_chunks]

            # Process results as they complete
            for future in futures:
                shift, score, zscore, early_termination = future.result()

                # If this result is better than what we have or we got an early termination
                if shift is not None:
                    if best_result is None or score > best_result[1]:
                        best_result = (shift, score, zscore)

                    # If we got an early termination, cancel all other futures
                    if early_termination:
                        logger.info(f"Excellent match found at shift {shift} with score {score:.4f} and z-score {zscore:.2f}")
                        for f in futures:
                            f.cancel()
                        break

        # If we found a result, return it
        if best_result:
            shift, score, zscore = best_result
            logger.info(f"Best match: shift={shift}, score={score:.4f}, z-score={zscore:.2f}")
            return shift, score, zscore

        # If no good match was found
        logger.info("No good match found")
        return None, 0, 0


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
