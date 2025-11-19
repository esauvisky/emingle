import os
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from skimage.feature import match_template
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import Config

class ImageMerger:
    @staticmethod
    def find_fixed_borders(images, margin=5):
        if not images: return 0, 0, 0, 0
        rgb_arrays = [np.array(img) for img in images]
        first_img = rgb_arrays[0]
        height, width, _ = first_img.shape
        top = 0
        bottom = height

        # Top Border
        for i in range(height // 2):
            row_is_fixed = True
            base_row = first_img[i, :, :]
            for other_img in rgb_arrays[1:]:
                if np.mean(np.abs(other_img[i, :, :] - base_row)) > margin:
                    row_is_fixed = False
                    break
            if not row_is_fixed:
                top = i
                break
        # Bottom Border
        for i in range(height - 1, height // 2, -1):
            row_is_fixed = True
            base_row = first_img[i, :, :]
            for other_img in rgb_arrays[1:]:
                if np.mean(np.abs(other_img[i, :, :] - base_row)) > margin:
                    row_is_fixed = False
                    break
            if not row_is_fixed:
                bottom = i + 1
                break
        if top >= bottom: top, bottom = 0, height
        return top, bottom, 0, width

    @staticmethod
    def remove_borders(array, top, bottom, left, right):
        return array[top:bottom, :]

    @staticmethod
    def compute_overlap_offset(base_img_arr, new_img_arr, debug_id=None, min_overlap=20, search_limit_ratio=1):
        def to_gray(arr):
            return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

        gray_base = to_gray(base_img_arr)
        gray_new = to_gray(new_img_arr)

        # 1. Edge Detection (Sobel)
        # This is naturally resistant to color changes, only caring about structure.
        feat_base = sobel(gray_base, axis=0)
        feat_new = sobel(gray_new, axis=0)

        h_base, w = feat_base.shape
        h_new, _ = feat_new.shape

        probe_height = int(h_new * 0.15)
        probe_height = max(probe_height, min_overlap)
        probe_height = min(probe_height, h_new - 1)

        probe = feat_new[:probe_height, :]

        search_height = int(h_base * search_limit_ratio)
        search_start_y = h_base - search_height
        search_region = feat_base[search_start_y:, :]

        # 2. Template Matching
        result = match_template(search_region, probe)
        ij = np.unravel_index(np.argmax(result), result.shape)
        y_match_local, x_match_local = ij
        match_score = result[y_match_local, x_match_local]

        shift = search_start_y + y_match_local
        overlap_height = h_base - shift

        if Config["DEBUG_MODE"]:
            ImageMerger._save_debug_plot(
                debug_id, gray_base, gray_new, feat_base, feat_new,
                result, shift, overlap_height, match_score, search_start_y
            )

        # Slightly lowered threshold to allow Sobel to find matches even if noise exists
        if match_score < 0.4:
            return None, 0

        return shift, overlap_height

    @staticmethod
    def validate_overlap_robust(base_arr, new_arr, shift, overlap_height, tolerance=15.0, ignore_worst_percent=0.20):
        """
        Validates overlap by checking if the MAJORITY of pixels match.

        Args:
            tolerance: The average pixel difference allowed (0-255).
                       15 is lenient enough for compression artifacts.
            ignore_worst_percent: The percentage of pixels to IGNORE.
                                  0.20 means we ignore the 20% most different pixels
                                  (blinking cursors, spinners, changing numbers).
        """
        if overlap_height <= 0: return False

        # Extract Overlapping Regions
        region_base = base_arr[shift : shift + overlap_height, :, :]
        region_new = new_arr[:overlap_height, :, :]

        if region_base.shape != region_new.shape: return False

        # Convert to Grayscale for simpler math
        def to_gray(arr): return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

        r1 = to_gray(region_base)
        r2 = to_gray(region_new)

        # Calculate Absolute Difference
        diff = np.abs(r1 - r2)

        # Flatten to 1D array
        flat_diff = diff.flatten()

        # Determine how many pixels to keep (e.g., keep best 80%)
        keep_count = int(flat_diff.size * (1.0 - ignore_worst_percent))

        if keep_count == 0: return False

        # Fast partial sort to find the smallest 'keep_count' errors
        # argpartition puts the smallest K elements at the front (unordered)
        # This is much faster than a full sort.
        partitioned_indices = np.argpartition(flat_diff, keep_count)
        best_pixels = flat_diff[partitioned_indices[:keep_count]]

        # Calculate the Mean Error of the "Good" pixels
        mean_error = np.mean(best_pixels)

        logger.debug(f"Robust Validation Error: {mean_error:.2f} (Tolerance: {tolerance})")

        # If the background and static text match (low error), we return True
        # regardless of what the blinking cursor (high error) is doing.
        return mean_error < tolerance

    @staticmethod
    def merge_images_vertically(base_img, new_img, debug_id=None):
        base_arr = np.array(base_img)
        new_arr = np.array(new_img)

        if base_arr.shape[1] != new_arr.shape[1]:
            return base_img

        shift, overlap_height = ImageMerger.compute_overlap_offset(base_arr, new_arr, debug_id=debug_id)

        if shift is None or overlap_height < 10:
            logger.warning(f"Step {debug_id}: No structure match. Appending.")
            return base_img
            # return Image.fromarray(np.vstack((base_arr, new_arr)))

        # Use the new Robust Validation
        is_valid = ImageMerger.validate_overlap_robust(base_arr, new_arr, shift, overlap_height)

        if not is_valid:
            logger.warning(f"Step {debug_id}: Robust validation failed. Appending.")
            return base_img
            # return Image.fromarray(np.vstack((base_arr, new_arr)))

        logger.info(f"Step {debug_id}: Merging w/ Shift: {shift}, Overlap: {overlap_height}")

        # Use a Hard Cut at the center of the overlap.
        # Why? Because if we blend, a blinking cursor becomes a ghost cursor.
        # If we hard cut, we pick one state (either cursor on or off), which looks cleaner.
        cut_point = int(overlap_height / 2)

        part_a = base_arr[:shift]
        part_b_1 = base_arr[shift : shift + cut_point] # From Base
        part_b_2 = new_arr[cut_point : overlap_height] # From New
        part_c = new_arr[overlap_height:]

        merged = np.vstack((part_a, part_b_1, part_b_2, part_c))
        return Image.fromarray(merged)

    @staticmethod
    def _save_debug_plot(debug_id, gray_base, gray_new, feat_base, feat_new, result, shift, overlap_height, score, search_start_y):
        # (Same visualization code as previous step, but ensuring backend safety)
        try:
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(2, 2)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(feat_base[-int(feat_base.shape[0]/2):], cmap='jet')
            ax1.set_title("Edge Features (Base)")

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(result, cmap='viridis')
            ax2.set_title(f"Match Score: {score:.2f}")

            ax3 = fig.add_subplot(gs[1, :])
            if overlap_height > 0:
                # Visualizing the Robust Difference
                r1 = gray_base[shift : shift + overlap_height]
                r2 = gray_new[:overlap_height]
                if r1.shape == r2.shape:
                    diff = np.abs(r1 - r2)
                    # highlight large diffs in red
                    ax3.imshow(diff, cmap='gray', vmin=0, vmax=50)
                    ax3.set_title(f"Difference Map (Darker = Better Match)")

            os.makedirs("debug_output", exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"debug_output/merge_step_{debug_id}.png")
            plt.close(fig)
        except Exception:
            pass
