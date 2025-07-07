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
    def find_fixed_borders(images, margin=2):
        """
        Identifies the top and bottom rows that are consistent across all images.
        Left and right borders are considered part of the content and are not cropped.
        """
        rgb_arrays = [np.array(img) for img in images]

        top, bottom = 0, rgb_arrays[0].shape[0]
        width = rgb_arrays[0].shape[1] # Keep full width

        # Find top fixed border
        for i in range(rgb_arrays[0].shape[0]):
            is_fixed_row = True
            for array in rgb_arrays[1:]: # Compare against the first image
                if not np.allclose(array[i, :, :], rgb_arrays[0][i, :, :], rtol=margin):
                    is_fixed_row = False
                    break
            if not is_fixed_row:
                top = i
                break

        # Find bottom fixed border
        for i in range(rgb_arrays[0].shape[0] - 1, -1, -1):
            is_fixed_row = True
            for array in rgb_arrays[1:]: # Compare against the first image
                if not np.allclose(array[i, :, :], rgb_arrays[0][i, :, :], rtol=margin):
                    is_fixed_row = False
                    break
            if not is_fixed_row:
                bottom = i + 1
                break
        
        # Ensure bottom is not less than top, and within bounds
        bottom = max(top, bottom)
        bottom = min(bottom, rgb_arrays[0].shape[0])

        return top, bottom, 0, width # Always return 0 and width for left/right

    @staticmethod
    def remove_borders(array, top, bottom, left, right):
        """
        Removes only top and bottom borders, preserving full width.
        The 'left' and 'right' parameters are ignored for actual slicing.
        """
        return array[top:bottom, :]

    @staticmethod
    def _downsample_array(array, factor):
        """Downsample a NumPy array using PIL for resizing."""
        if factor == 1:
            return array
        height, width = array.shape
        new_height, new_width = height // factor, width // factor
        img = Image.fromarray(array)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(img_resized)

    @staticmethod
    def _process_shift_chunk(base_array_gray, new_array_gray, shift_chunk):
        """Process a chunk of shifts in a separate process to find best SSIM score."""
        from skimage.metrics import structural_similarity as ssim

        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        best_shift = None
        best_score = -1.0 # Initialize with lowest possible SSIM score

        results_in_chunk = [] # Store all (shift, score) pairs from this chunk

        for shift in shift_chunk:
            if shift >= 0:
                # Overlapping regions (new image below base image)
                overlap_height = min(height_base - shift, height_new)
                # base_overlap: from 'shift' downwards in base_array_gray
                # new_overlap: from top (0) downwards in new_array_gray
                base_overlap = base_array_gray[shift:shift + overlap_height, :]
                new_overlap = new_array_gray[:overlap_height, :]
            else:
                # Negative shift: new image above base image
                overlap_height = min(height_new + shift, height_base)
                # base_overlap: from top (0) downwards in base_array_gray
                # new_overlap: from '-shift' downwards in new_array_gray
                base_overlap = base_array_gray[:overlap_height, :]
                new_overlap = new_array_gray[-shift:-shift + overlap_height, :]

            # Ensure the overlapping regions have the same dimensions
            if base_overlap.shape != new_overlap.shape:
                continue

            # Skip if overlap is too small (needs at least 7 pixels for SSIM to be meaningful)
            if overlap_height < 7:
                continue

            # Calculate SSIM for structural similarity
            ssim_score = ssim(base_overlap, new_overlap, data_range=255)

            # Early termination if we find an exceptionally good match (e.g., almost perfect)
            if ssim_score > 0.995: # Very high score, likely the match.
                return (shift, ssim_score), True # Return (shift, score) tuple and True for early termination

            results_in_chunk.append((shift, ssim_score))

            if ssim_score > best_score:
                best_score = ssim_score
                best_shift = shift

        # Return all results from this chunk, and False for early termination
        return results_in_chunk, False

    @staticmethod
    def find_image_overlap(base_array_gray, new_array_gray, threshold=0.8, z_score_threshold=2.0):
        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        # Minimum valid overlap height for SSIM calculation
        min_overlap_height = 7

        fig = None
        axs = []
        if Config["DEBUG_MODE"]:
            fig, axs = plt.subplots(1, 3, figsize=(10, 8))  # Create three subplots side by side

        num_processes = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Using {num_processes} processes for parallel processing")

        # --- Step 1: Rough Search on Downsampled Images ---
        downsample_factor = 4  # A reasonable factor for initial speedup
        logger.info(f"Performing rough search on downsampled images (factor: {downsample_factor})...")

        base_array_gray_downsampled = ImageMerger._downsample_array(base_array_gray, downsample_factor)
        new_array_gray_downsampled = ImageMerger._downsample_array(new_array_gray, downsample_factor)

        height_base_ds, _ = base_array_gray_downsampled.shape
        height_new_ds, _ = new_array_gray_downsampled.shape

        # Define the full range for shifts for downsampled images
        min_shift_ds = -(height_new_ds - min_overlap_height)
        max_shift_ds = (height_base_ds - min_overlap_height)
        shift_range_ds = range(min_shift_ds, max_shift_ds + 1)

        if not shift_range_ds:
            logger.warning("Rough search shift range is empty. Returning no match.")
            return None, 0, 0

        chunk_size_ds = max(1, len(shift_range_ds) // num_processes)
        shift_chunks_ds = [shift_range_ds[i:i + chunk_size_ds] for i in range(0, len(shift_range_ds), chunk_size_ds)]

        all_results_ds = []
        early_termination_rough = False

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures_ds = [executor.submit(
                ImageMerger._process_shift_chunk,
                base_array_gray_downsampled,
                new_array_gray_downsampled,
                chunk
            ) for chunk in shift_chunks_ds]

            for future_ds in futures_ds:
                chunk_results, early_termination_flag = future_ds.result()
                if early_termination_flag:
                    # If any worker found an excellent match, stop immediately
                    best_shift, best_score = chunk_results # When early_termination_flag is True, chunk_results is (shift, score)
                    logger.info(f"Early termination during rough search. Shift: {best_shift} (ds), Score: {best_score:.4f}")
                    # Convert to original scale and mark for final decision
                    rough_shift_original = best_shift * downsample_factor
                    return rough_shift_original, best_score, float('inf') # Use inf for z-score to guarantee selection
                all_results_ds.extend(chunk_results)

        if not all_results_ds:
            logger.info("No results found during rough search. Returning no match.")
            return None, 0, 0

        # Calculate Z-scores for all collected rough search results
        shifts_ds = np.array([res[0] for res in all_results_ds])
        scores_ds = np.array([res[1] for res in all_results_ds])

        # Filter results based on initial SSIM score
        filtered_indices_ds = scores_ds >= threshold
        if not np.any(filtered_indices_ds):
            logger.info("No rough match found above SSIM threshold. Returning no match.")
            return None, 0, 0

        filtered_scores_ds = scores_ds[filtered_indices_ds]
        filtered_shifts_ds = shifts_ds[filtered_indices_ds]

        if len(filtered_scores_ds) < 5: # Need enough data points for meaningful std/z-score
            # If not enough data for z-score, pick the best score among filtered
            best_idx_ds = np.argmax(filtered_scores_ds)
            best_shift_ds = filtered_shifts_ds[best_idx_ds]
            best_score_ds = filtered_scores_ds[best_idx_ds]
            best_zscore_ds = 0 # No meaningful z-score
            logger.info(f"Rough search: Not enough data for robust Z-score, best score selected based on SSIM. Shift: {best_shift_ds} (ds), Score: {best_score_ds:.4f}")
        else:
            mean_score_ds = np.mean(filtered_scores_ds)
            std_score_ds = np.std(filtered_scores_ds)

            z_scores_ds = (filtered_scores_ds - mean_score_ds) / (std_score_ds + 1e-8) # Add epsilon to prevent division by zero

            # Find the best result that meets the z-score threshold
            z_score_filtered_indices_ds = z_scores_ds > z_score_threshold
            if not np.any(z_score_filtered_indices_ds):
                logger.info("No rough match found above Z-score threshold. Returning no match.")
                return None, 0, 0

            # Select the best among the z-score filtered results
            best_idx_ds = np.argmax(filtered_scores_ds[z_score_filtered_indices_ds])
            best_shift_ds = filtered_shifts_ds[z_score_filtered_indices_ds][best_idx_ds]
            best_score_ds = filtered_scores_ds[z_score_filtered_indices_ds][best_idx_ds]
            best_zscore_ds = z_scores_ds[z_score_filtered_indices_ds][best_idx_ds]
            logger.info(f"Rough shift found: {best_shift_ds} (downsampled), Score: {best_score_ds:.4f}, Z-score: {best_zscore_ds:.2f}")


        # --- Step 2: Refinement Search on Original Images ---
        rough_shift_original = best_shift_ds * downsample_factor
        logger.info(f"Rough shift found: {best_shift_ds} (downsampled), {rough_shift_original} (original scale).")

        # Define a refinement window around the rough shift
        refinement_window_size = downsample_factor * 15 # A slightly larger window for refinement
        min_shift_refined = max(-(height_new - min_overlap_height), rough_shift_original - refinement_window_size)
        max_shift_refined = min((height_base - min_overlap_height), rough_shift_original + refinement_window_size)

        shift_range_refined = range(min_shift_refined, max_shift_refined + 1)
        if not shift_range_refined:
            logger.warning("Refinement shift range is empty. Returning no match.")
            return None, 0, 0

        logger.info(f"Performing refinement search on original images in range: [{min_shift_refined}, {max_shift_refined}].")
        chunk_size_refined = max(1, len(shift_range_refined) // num_processes)
        shift_chunks_refined = [shift_range_refined[i:i + chunk_size_refined] for i in range(0, len(shift_range_refined), chunk_size_refined)]

        all_results_refined = []
        early_termination_refined = False

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures_refined = [executor.submit(
                ImageMerger._process_shift_chunk,
                base_array_gray,  # Use original arrays
                new_array_gray,   # Use original arrays
                chunk
            ) for chunk in shift_chunks_refined]

            for future_refined in futures_refined:
                chunk_results, early_termination_flag = future_refined.result()
                if early_termination_flag:
                    # If any worker found an excellent match, stop immediately
                    best_shift, best_score = chunk_results
                    logger.info(f"Excellent match during refinement search. Shift: {best_shift}, Score: {best_score:.4f}")
                    return best_shift, best_score, float('inf') # Use inf for z-score to guarantee selection
                all_results_refined.extend(chunk_results)

        if not all_results_refined:
            logger.info("No results found during refinement search. Returning no match.")
            return None, 0, 0

        # Calculate Z-scores for all collected refinement search results
        shifts_refined = np.array([res[0] for res in all_results_refined])
        scores_refined = np.array([res[1] for res in all_results_refined])

        # Filter results based on initial SSIM score
        filtered_indices_refined = scores_refined >= threshold
        if not np.any(filtered_indices_refined):
            logger.info("No refined match found above SSIM threshold. Returning no match.")
            return None, 0, 0

        filtered_scores_refined = scores_refined[filtered_indices_refined]
        filtered_shifts_refined = shifts_refined[filtered_indices_refined]

        if len(filtered_scores_refined) < 5: # Need enough data points for meaningful std/z-score
            best_idx_refined = np.argmax(filtered_scores_refined)
            shift = filtered_shifts_refined[best_idx_refined]
            score = filtered_scores_refined[best_idx_refined]
            zscore = 0 # No meaningful z-score
            logger.info(f"Refinement search: Not enough data for robust Z-score, best score selected based on SSIM. Shift: {shift}, Score: {score:.4f}")
        else:
            mean_score_refined = np.mean(filtered_scores_refined)
            std_score_refined = np.std(filtered_scores_refined)

            z_scores_refined = (filtered_scores_refined - mean_score_refined) / (std_score_refined + 1e-8)

            # Find the best result that meets the z-score threshold
            z_score_filtered_indices_refined = z_scores_refined > z_score_threshold
            if not np.any(z_score_filtered_indices_refined):
                logger.info("No refined match found above Z-score threshold. Returning no match.")
                return None, 0, 0

            # Select the best among the z-score filtered results
            best_idx_refined = np.argmax(filtered_scores_refined[z_score_filtered_indices_refined])
            shift = filtered_shifts_refined[z_score_filtered_indices_refined][best_idx_refined]
            score = filtered_scores_refined[z_score_filtered_indices_refined][best_idx_refined]
            zscore = z_scores_refined[z_score_filtered_indices_refined][best_idx_refined]
            logger.info(f"Best match (refined): shift={shift}, score={score:.4f}, z-score={zscore:.2f}")

        return shift, score, zscore


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
    @staticmethod
    def add_borders_back(cropped_merged_image, original_first_screenshot_array, top_border_height, left_border_width, right_border_width, original_bottom_index):
        """
        Adds the fixed borders back to a merged, cropped image.
        Args:
            cropped_merged_image (PIL.Image): The merged image without fixed borders.
            original_first_screenshot_array (np.array): The NumPy array of the very first original screenshot (full image).
            top_border_height (int): The height of the fixed top border (from find_fixed_borders 'top' value).
            left_border_width (int): The width of the fixed left border (from find_fixed_borders 'left' value).
            right_border_width (int): The width of the fixed right border (calculated as original_width - 'right' value).
            original_bottom_index (int): The row index (exclusive) of the end of the common content area
                                        in the original screenshot (from find_fixed_borders 'bottom' value).
        Returns:
            PIL.Image: The final merged image with fixed borders re-added.
        """
        cropped_merged_array = np.array(cropped_merged_image)
        merged_height, merged_width, channels = cropped_merged_array.shape

        original_width_src = original_first_screenshot_array.shape[1]

        final_width = left_border_width + merged_width + right_border_width
        final_height = top_border_height + merged_height

        final_array = np.zeros((final_height, final_width, channels), dtype=np.uint8)

        # 1. Place the top border
        if top_border_height > 0:
            # Copy the top_border_height rows from the original first screenshot
            # The width of the copied segment should be the full final_width (which equals original_width_src)
            final_array[0:top_border_height, 0:final_width] = original_first_screenshot_array[0:top_border_height, 0:final_width]

        # 2. Place the merged (cropped) content
        final_array[top_border_height : top_border_height + merged_height,
                    left_border_width : left_border_width + merged_width] = cropped_merged_array

        # 3. Place left border
        if left_border_width > 0:
            # Get a sample strip of the left border from the original image's content area
            # This strip will be tiled vertically to cover the entire merged height.
            left_border_sample = original_first_screenshot_array[
                top_border_height:original_bottom_index,
                0:left_border_width
            ]
            if left_border_sample.shape[0] > 0:
                tiled_left_border = np.tile(left_border_sample, (int(np.ceil(merged_height / left_border_sample.shape[0])), 1, 1))
                tiled_left_border = tiled_left_border[:merged_height, :, :]
                final_array[top_border_height : top_border_height + merged_height, 0 : left_border_width] = tiled_left_border
            else:
                logger.warning("Left border sample is empty, cannot add left border.")

        # 4. Place right border
        if right_border_width > 0:
            # Get a sample strip of the right border from the original image's content area
            right_border_sample = original_first_screenshot_array[
                top_border_height:original_bottom_index,
                original_width_src - right_border_width : original_width_src
            ]
            if right_border_sample.shape[0] > 0:
                tiled_right_border = np.tile(right_border_sample, (int(np.ceil(merged_height / right_border_sample.shape[0])), 1, 1))
                tiled_right_border = tiled_right_border[:merged_height, :, :]
                final_array[top_border_height : top_border_height + merged_height,
                            left_border_width + merged_width : final_width] = tiled_right_border
            else:
                logger.warning("Right border sample is empty, cannot add right border.")

        return Image.fromarray(final_array)

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
