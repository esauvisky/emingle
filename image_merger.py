
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
    def _calculate_static_bounds(arr1, arr2, threshold=10):
        """
        Compares the top and bottom of two images to find static UI elements
        (Headers, Footers, Navigation bars) that shouldn't be part of matching.
        """
        if arr1.shape != arr2.shape:
            # If shapes differ (base grew), we can only compare the 'original' screen size area.
            # But simpler: compare row by row until difference is high.
            h = min(arr1.shape[0], arr2.shape[0])
        else:
            h = arr1.shape[0]

        # 1. Top Static Region (Header)
        top_static = 0
        for i in range(0, h // 3): # Check top 1/3 max
            # Compare row i of Base (Top) vs row i of New (Top)
            # Use arr1[i] vs arr2[i].
            # Note: For a stitched Base, arr1[0] is the original header.
            diff = np.mean(np.abs(arr1[i, :, :] - arr2[i, :, :]))
            if diff < threshold:
                top_static = i + 1
            else:
                break

        # 2. Bottom Static Region (Footer)
        bottom_static = 0
        for i in range(1, h // 3):
            # Compare bottom up
            base_idx = arr1.shape[0] - i
            new_idx = arr2.shape[0] - i

            diff = np.mean(np.abs(arr1[base_idx, :, :] - arr2[new_idx, :, :]))
            if diff < threshold:
                bottom_static = i
            else:
                break

        return top_static, bottom_static

    @staticmethod
    def compute_overlap_offset(base_img_arr, new_img_arr, crop_top=0, crop_bottom=0, debug_id=None, min_overlap=20, search_limit_ratio=1):
        def to_gray(arr):
            return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])

        gray_base = to_gray(base_img_arr)
        gray_new = to_gray(new_img_arr)

        # 1. Edge Detection
        feat_base = sobel(gray_base, axis=0)
        feat_new = sobel(gray_new, axis=0)

        h_base, w = feat_base.shape
        h_new, _ = feat_new.shape

        # --- KEY FIX: Define Probe based on DYNAMIC content only ---
        # We start the probe AFTER the static header.
        # We end the probe BEFORE the static footer.

        effective_h_new = h_new - crop_top - crop_bottom
        if effective_h_new < 50:
            # Fallback if too much is cropped
            crop_top = 0
            crop_bottom = 0

        # Probe size: 15% of the *dynamic* area
        probe_height = int(effective_h_new * 0.15)
        probe_height = max(probe_height, min_overlap)

        # Probe start: Below header
        probe_y_start = crop_top
        probe_y_end = probe_y_start + probe_height

        if probe_y_end >= (h_new - crop_bottom):
             probe_y_end = h_new - crop_bottom - 1

        probe = feat_new[probe_y_start:probe_y_end, :]

        # Search Region: Bottom of base image
        search_height = int(h_base * search_limit_ratio)
        search_start_y = h_base - search_height
        search_region = feat_base[search_start_y:, :]

        # 2. Template Matching
        result = match_template(search_region, probe)
        ij = np.unravel_index(np.argmax(result), result.shape)
        y_match_local, x_match_local = ij
        match_score = result[y_match_local, x_match_local]

        # Map back to global coordinates
        # The match found where 'probe' fits.
        # Shift = where the top of New Image fits into Base.
        # matched_y_in_search = y_match_local
        # matched_y_in_base = search_start_y + y_match_local
        # This matched_y corresponds to the start of the PROBE (which is at crop_top).
        # So the top of the NEW image (0) is at matched_y - crop_top.

        match_y_global = search_start_y + y_match_local
        shift = match_y_global - probe_y_start

        overlap_height = h_base - shift

        if Config["DEBUG_MODE"]:
            ImageMerger._save_debug_plot(
                debug_id, gray_base, gray_new, feat_base, feat_new,
                result, shift, overlap_height, match_score, search_start_y,
                probe_y_start, probe_height
            )

        if match_score < 0.4:
            return None, 0

        return shift, overlap_height

    @staticmethod
    def validate_overlap_robust(base_arr, new_arr, shift, overlap_height, crop_top=0, crop_bottom=0, tolerance=20.0):
        if overlap_height <= 0: return False

        # Extract Overlapping Regions
        region_base = base_arr[shift : shift + overlap_height, :, :]
        region_new = new_arr[:overlap_height, :, :]

        if region_base.shape != region_new.shape: return False

        # --- KEY FIX: Mask out Static Header/Footer from Validation ---
        # If the overlap includes the top of New Image, we must ignore the header
        # because the Base Image might not have that header at that specific location
        # (unless Shift==0, which we are trying to avoid).

        # Actually, simpler logic:
        # We are validating if pixels match.
        # If we overlap the Header area, and the Header is static, it WILL match.
        # But we want to know if the *content* matches.
        # So we slice off the static zones from the calculation.

        valid_start = 0
        valid_end = region_new.shape[0]

        # If overlap covers the top of New Image, mask the header
        if valid_start < crop_top:
            valid_start = crop_top

        # If overlap covers the bottom of New Image, mask the footer
        # (Though usually overlap is at the top of New)
        # Note: crop_bottom is from the bottom edge.
        limit_bottom = new_arr.shape[0] - crop_bottom
        if overlap_height > limit_bottom:
            valid_end = limit_bottom

        # Safety: if we cropped everything (overlap is purely inside the header?), fail.
        if valid_end <= valid_start:
            # If the overlap is ENTIRELY inside the header, it's ambiguous.
            # But usually we want to merge content.
            # If Shift is massive (small overlap) and we are only overlapping the header,
            # we might return True (technically matches) or False (wait for more content).
            # Let's be safe and allow it if it's strictly the header, but usually we want content.
            return True

        # Slice to valid dynamic area
        v_base = region_base[valid_start:valid_end]
        v_new = region_new[valid_start:valid_end]

        def to_gray(arr): return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        r1 = to_gray(v_base)
        r2 = to_gray(v_new)

        sx1, sy1 = sobel(r1, axis=0), sobel(r1, axis=1)
        sx2, sy2 = sobel(r2, axis=0), sobel(r2, axis=1)
        mag1 = np.hypot(sx1, sy1)
        mag2 = np.hypot(sx2, sy2)

        content_mask = (mag1 > 30) | (mag2 > 30)

        diff = np.abs(r1 - r2)

        total_content_pixels = np.sum(content_mask)
        total_pixels = r1.size
        content_ratio = total_content_pixels / total_pixels

        if content_ratio < 0.01:
            mean_error = np.mean(diff)
            return mean_error < 5.0

        content_diff = diff[content_mask]
        median_content_error = np.median(content_diff)

        logger.debug(f"Validation (Dynamic Area {valid_start}-{valid_end}): Median Err={median_content_error:.2f}")

        return median_content_error < tolerance

    @staticmethod
    def merge_images_vertically(base_img, new_img, debug_id=None, tolerance=20.0):
        base_arr = np.array(base_img)
        new_arr = np.array(new_img)

        if base_arr.shape[1] != new_arr.shape[1]:
            return base_img

        # 1. Detect Static Bars
        t_crop, b_crop = ImageMerger._calculate_static_bounds(base_arr, new_arr)
        if t_crop > 0 or b_crop > 0:
            logger.debug(f"Detected static bars: Top {t_crop}px, Bottom {b_crop}px")

        # 2. Compute Offset (ignoring static bars)
        shift, overlap_height = ImageMerger.compute_overlap_offset(
            base_arr, new_arr,
            crop_top=t_crop,
            crop_bottom=b_crop,
            debug_id=debug_id
        )

        if shift is None or overlap_height < 10:
            logger.warning(f"Step {debug_id}: No structure match. Appending.")
            metadata = {
                'static_top': t_crop,
                'static_bottom': b_crop,
                'overlap_height': 0,
                'shift': None
            }
            return base_img, metadata

        # 3. Validate (ignoring static bars)
        is_valid = ImageMerger.validate_overlap_robust(
            base_arr, new_arr, shift, overlap_height,
            crop_top=t_crop, crop_bottom=b_crop, tolerance=tolerance
        )

        if not is_valid:
            logger.warning(f"Step {debug_id}: Validation failed.")
            metadata = {
                'static_top': t_crop,
                'static_bottom': b_crop,
                'overlap_height': overlap_height,
                'shift': shift
            }
            return base_img, metadata

        logger.info(f"Step {debug_id}: Merging w/ Shift: {shift}, Overlap: {overlap_height}")

        cut_point = int(overlap_height / 2)

        part_a = base_arr[:shift]
        part_b_1 = base_arr[shift : shift + cut_point]
        part_b_2 = new_arr[cut_point : overlap_height]
        part_c = new_arr[overlap_height:]

        merged = np.vstack((part_a, part_b_1, part_b_2, part_c))
        metadata = {
            'static_top': t_crop,
            'static_bottom': b_crop,
            'overlap_height': overlap_height,
            'shift': shift
        }
        return Image.fromarray(merged), metadata

    @staticmethod
    def _save_debug_plot(debug_id, gray_base, gray_new, feat_base, feat_new, result, shift, overlap_height, score, search_start_y, probe_y=0, probe_h=0):
        try:
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

            # Row 1: Original Images
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(gray_base, cmap='gray')
            ax1.set_title(f"Base Image\n{gray_base.shape}")
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(gray_new, cmap='gray')
            # Draw probe box on New Image
            if probe_h > 0:
                rect = patches.Rectangle((0, probe_y), gray_new.shape[1], probe_h, linewidth=2, edgecolor='cyan', facecolor='none')
                ax2.add_patch(rect)
            ax2.set_title(f"New Image (Probe Cyan)\n{gray_new.shape}")
            ax2.axis('off')

            ax3 = fig.add_subplot(gs[0, 2])
            if overlap_height > 0:
                r1 = gray_base[shift : shift + overlap_height]
                r2 = gray_new[:overlap_height]
                if r1.shape == r2.shape:
                    diff = np.abs(r1 - r2)
                    im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=50)
                    ax3.set_title(f"Diff (Top of New)")
                    plt.colorbar(im3, ax=ax3, shrink=0.6)
            ax3.axis('off')

            # Row 2: Sobel
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(feat_base, cmap='RdBu_r', vmin=-50, vmax=50)
            ax4.set_title("Base Edges")
            ax4.axis('off')

            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(feat_new, cmap='RdBu_r', vmin=-50, vmax=50)
            ax5.set_title("New Edges")
            ax5.axis('off')

            # Row 3: Result
            ax7 = fig.add_subplot(gs[2, 0])
            im7 = ax7.imshow(result, cmap='viridis')
            ax7.set_title(f"Match Result\nScore: {score:.3f}")
            plt.colorbar(im7, ax=ax7, shrink=0.6)

            ax8 = fig.add_subplot(gs[2, 1])
            search_region = feat_base[search_start_y:, :]
            ax8.imshow(search_region, cmap='gray')
            if overlap_height > 0:
                # Calculate where the probe matched within the search region
                match_y_in_search = shift + probe_y - search_start_y
                rect = patches.Rectangle((0, match_y_in_search), search_region.shape[1], probe_h,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax8.add_patch(rect)
            ax8.set_title(f"Search Region (Match Red)")
            ax8.axis('off')

            ax9 = fig.add_subplot(gs[2, 2])
            stats = f"Shift: {shift}\nOverlap: {overlap_height}\nProbe Y: {probe_y}\nProbe H: {probe_h}"
            ax9.text(0.1, 0.5, stats, transform=ax9.transAxes)
            ax9.axis('off')

            os.makedirs("debug_output", exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"debug_output/merge_step_{debug_id}.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass
