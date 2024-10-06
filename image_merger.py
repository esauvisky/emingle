import time
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image
from loguru import logger


class ImageMerger:
    @staticmethod
    def find_image_overlap(base_array, new_array, threshold=0.995):
        base_array_tuple = tuple(map(tuple, base_array))
        new_array_tuple = tuple(map(tuple, new_array))
        base_array = np.array(base_array_tuple)
        new_array = np.array(new_array_tuple)

        logger.debug(f"Entering find_image_overlap with threshold {threshold}")
        start_time = time.time()

        height_base, width, _ = base_array.shape
        height_new = new_array.shape[0]

        # Vectorized comparison
        shifts = np.arange(-height_new + 1, height_base)
        match_percentages = np.zeros_like(shifts, dtype=float)

        for i, shift in enumerate(shifts):
            start_base = max(0, shift)
            end_base = min(height_base, shift + height_new)
            start_new = max(0, -shift)
            end_new = start_new + (end_base-start_base)

            if end_base - start_base > 0:
                base_overlap = base_array[start_base:end_base]
                new_overlap = new_array[start_new:end_new]

                # Calculates which rows from the new image match the
                # rows from the base image within the overlapping part
                # Use axis=(2) to compare pixel by pixel instead row by row
                matching_rows = np.all(base_overlap == new_overlap, axis=(1, 2))

                match_percentages[i] = np.mean(matching_rows)

        best_match_index = np.argmax(match_percentages)
        best_match_percentage = match_percentages[best_match_index]
        best_shift = shifts[best_match_index] if best_match_percentage >= threshold else None

        end_time = time.time()
        logger.debug(f"Exiting find_image_overlap. Time taken: {end_time - start_time:.4f} seconds")
        return best_shift, best_match_percentage

    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.995):
        logger.debug("Entering merge_images_vertically")
        base_array = np.array(base_img)
        new_array = np.array(new_img)

        if base_img.mode != new_img.mode:
            raise ValueError(f"Image modes don't match: {base_img.mode} vs {new_img.mode}")

        if base_array.shape[1] != new_array.shape[1]:
            raise ValueError("Images must have the same width")

        shift, match_percentage = ImageMerger.find_image_overlap(base_array, new_array, threshold)

        if shift is not None:
            logger.info(f"Overlap detected at shift {shift} with match percentage {match_percentage:.2%}")

            height_base, width, _ = base_array.shape
            height_new = new_array.shape[0]

            if shift < 0:
                merged_array = np.vstack((new_array[0:-shift], base_array))
            else:
                overlap_size = min(height_new, height_base - shift)
                non_overlap_new = new_array[overlap_size:]
                merged_array = np.vstack((base_array, non_overlap_new)) if non_overlap_new.size > 0 else base_array

            return Image.fromarray(merged_array)
        else:
            logger.info("No overlap detected above the threshold, returning original base image")
            return base_img

    @staticmethod
    def process_single_image(args):
        base_img_path, new_img_path, threshold = args
        try:
            base_img = Image.open(base_img_path)
            new_img = Image.open(new_img_path)

            result = ImageMerger.merge_images_vertically(base_img, new_img, threshold)
            return result
        except Exception as e:
            logger.error(f"Failed to merge images: {str(e)}")
            return None

    @staticmethod
    def process_images(base_img_paths, new_img_paths, threshold=0.995, num_processes=4):
        logger.info("Entering process_images")
        start_time = time.time()
        with Pool(num_processes) as pool:
            results = pool.map(ImageMerger.process_single_image, [(base_path, new_path, threshold)
                                                                    for base_path, new_path in zip(base_img_paths, new_img_paths)]) # yapf: disable

        successful_merges = [result for result in results if result is not None]

        end_time = time.time()
        logger.info(f"Images merged successfully. Time taken: {end_time - start_time:.4f} seconds")

        return successful_merges

    find_image_overlap = find_image_overlap


if __name__ == "__main__":
    base_paths = ["test/2_base.png", "test/1_base.png", "test/3_base.png", "test/3_base.png", "test/3_base.png"]
    new_paths = ["test/2_new.png", "test/1_new.png", "test/3_new_1.png", "test/3_new_2.png", "test/3_new_3.png"]
    results = ImageMerger.process_images(base_paths, new_paths, threshold=0)
    # for result in results:
    #     result.show()
