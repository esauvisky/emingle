import os
import numpy as np
from PIL import Image
from skimage.feature import match_template
from loguru import logger
import matplotlib.pyplot as plt
from utils import Config

class ImageMerger:
    @staticmethod
    def find_fixed_borders(images, margin=5):
        if not images: return 0, 0, 0, 0
        rgb_arrays = [np.array(img) for img in images]
        first_img = rgb_arrays[0]
        height, width, _ = first_img.shape
        top, bottom = 0, height

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
    def compute_overlap_offset(base_img_arr, new_img_arr, debug_id=None, min_overlap=50):
        def to_gray(arr):
            return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

        gray_base = to_gray(base_img_arr)
        gray_new = to_gray(new_img_arr)

        # Use Grayscale (Good for text/code, specific characters)
        # Avoid Sobel here as it confuses identical code indentations
        feat_base = gray_base
        feat_new = gray_new

        h_base, w = feat_base.shape
        h_new, _ = feat_new.shape

        # Probe: Top 25% of new image
        probe_height = int(h_new * 0.25)
        probe_height = max(probe_height, min_overlap)
        probe_height = min(probe_height, h_new - 1)

        probe = feat_new[:probe_height, :]

        # --- CHANGE: Search the ENTIRE base image ---
        # We no longer limit to the bottom X%.
        # This allows for very small scrolls (huge overlaps).
        search_region = feat_base
        search_start_y = 0

        # Template Matching
        result = match_template(search_region, probe)
        ij = np.unravel_index(np.argmax(result), result.shape)
        y_match_local, x_match_local = ij
        match_score = result[y_match_local, x_match_local]

        # Offset Calculation
        shift = search_start_y + y_match_local
        overlap_height = h_base - shift

        if Config["DEBUG_MODE"]:
            # For visualization, if the base is huge, we still only show the
            # relevant bottom part so the image isn't squashed.
            ImageMerger._save_debug_plot(
                debug_id, gray_base, gray_new, feat_base, feat_new,
                result, shift, overlap_height, match_score
            )

        if match_score < 0.8:
            return None, 0

        return shift, overlap_height

    @staticmethod
    def validate_overlap_robust(base_arr, new_arr, shift, overlap_height, tolerance=10.0, ignore_worst_percent=0.20):
        if overlap_height <= 0: return False

        region_base = base_arr[shift : shift + overlap_height, :, :]
        region_new = new_arr[:overlap_height, :, :]

        if region_base.shape != region_new.shape: return False

        def to_gray(arr): return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

        diff = np.abs(to_gray(region_base) - to_gray(region_new)).flatten()
        keep_count = int(diff.size * (1.0 - ignore_worst_percent))

        if keep_count == 0: return False

        partitioned = np.argpartition(diff, keep_count)
        mean_error = np.mean(diff[partitioned[:keep_count]])

        logger.debug(f"Validation Error: {mean_error:.2f} (Tol: {tolerance})")
        return mean_error < tolerance

    @staticmethod
    def merge_images_vertically(base_img, new_img, debug_id=None):
        base_arr = np.array(base_img)
        new_arr = np.array(new_img)

        if base_arr.shape[1] != new_arr.shape[1]:
            return base_img

        shift, overlap_height = ImageMerger.compute_overlap_offset(base_arr, new_arr, debug_id=debug_id)

        if shift is None:
            logger.warning(f"Step {debug_id}: No match found (Score too low). Returning original base_img.")
            return base_img
            # return Image.fromarray(np.vstack((base_arr, new_arr)))

        is_valid = ImageMerger.validate_overlap_robust(base_arr, new_arr, shift, overlap_height)

        if not is_valid:
            logger.warning(f"Step {debug_id}: Validation failed. Returning original base_img.")
            return base_img
            # return Image.fromarray(np.vstack((base_arr, new_arr)))

        logger.info(f"Step {debug_id}: Merging w/ Shift: {shift}, Overlap: {overlap_height}")

        cut_point = int(overlap_height / 2)
        part_a = base_arr[:shift]
        part_b_1 = base_arr[shift : shift + cut_point]
        part_b_2 = new_arr[cut_point : overlap_height]
        part_c = new_arr[overlap_height:]

        merged = np.vstack((part_a, part_b_1, part_b_2, part_c))
        return Image.fromarray(merged)

    @staticmethod
    def _save_debug_plot(debug_id, gray_base, gray_new, feat_base, feat_new, result, shift, overlap_height, score):
        try:
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 2)

            # Display the bottom 1000px of the base image features to keep it readable
            # (The algo searched the whole thing, but we zoom in for the human)
            display_height = min(feat_base.shape[0], 1000)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(feat_base[-display_height:], cmap='gray')
            ax1.set_title("Features (Base - Bottom Zoom)")

            # Display the heatmap matching the zoom level
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(result[-display_height:], cmap='viridis')
            ax2.set_title(f"Match Score: {score:.2f}")

            ax3 = fig.add_subplot(gs[1, :])
            if overlap_height > 0:
                r1 = gray_base[shift : shift + overlap_height]
                r2 = gray_new[:overlap_height]
                if r1.shape == r2.shape:
                    diff = np.abs(r1 - r2)
                    ax3.imshow(diff, cmap='gray', vmin=0, vmax=50)
                    ax3.set_title(f"Difference Map (Dark=Match)")

            os.makedirs("debug_output", exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"debug_output/merge_step_{debug_id}.png")
            plt.close(fig)
        except Exception:
            pass
