import filecmp
import glob
import os
import time

from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from utils import Config


class ImageMerger:
    @staticmethod
    def find_fixed_borders(images, margin=0):
        # Convert all images to grayscale numpy arrays
        grayscale_arrays = [np.array(img.convert('L')) for img in images]

        # Initialize borders
        top, bottom, left, right = 0, grayscale_arrays[0].shape[0], 0, grayscale_arrays[0].shape[1]

        # Check each side for fixed borders
        for array in grayscale_arrays:
            # Check top border
            for i in range(top, array.shape[0]):
                if not np.allclose(array[i, :], grayscale_arrays[0][i, :], atol=margin):
                    top = i
                    break

            # Check bottom border
            for i in range(bottom - 1, -1, -1):
                if not np.allclose(array[i, :], grayscale_arrays[0][i, :], atol=margin):
                    bottom = i + 1
                    break

            # Check left border
            for i in range(left, array.shape[1]):
                if not np.allclose(array[:, i], grayscale_arrays[0][:, i], atol=margin):
                    left = i
                    break

            # Check right border
            for i in range(right - 1, -1, -1):
                if not np.allclose(array[:, i], grayscale_arrays[0][:, i], atol=margin):
                    right = i + 1
                    break

        return top, bottom, left, right

    @staticmethod
    def remove_borders(array, top, bottom, left, right):
        return array[top:bottom, left:right]

    @staticmethod
    def find_image_overlap(base_array_gray, new_array_gray, threshold=0.5):
        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        best_shift = None
        best_score = np.inf # Initialize with a large value

        fig = None
        axs = []
        if Config["DEBUG_MODE"]:
            fig, axs = plt.subplots(1, 3, figsize=(10, 5)) # Create three subplots side by side

        shifts = []
        match_percentages = []

        # We will consider both positive and negative shifts
        margin = height_new // 10
        for shift in range(-height_new + 1 + margin, height_base - margin):
            if shift >= 0:
                # Overlapping regions
                overlap_height = min(height_base - shift, height_new)
                if overlap_height <= 0:
                    continue
                base_overlap = base_array_gray[shift:shift + overlap_height, :]
                new_overlap = new_array_gray[:overlap_height, :]
            else:
                # Negative shift: new image is shifted down
                overlap_height = min(height_new + shift, height_base)
                if overlap_height <= 0:
                    continue
                base_overlap = base_array_gray[:overlap_height, :]
                new_overlap = new_array_gray[-shift:-shift + overlap_height, :]

            # Ensure the overlapping regions have the same dimensions
            if base_overlap.shape != new_overlap.shape:
                continue

            # Compute Sum of Absolute Differences
            sad = np.sum(np.abs(base_overlap.astype(np.int16) - new_overlap.astype(np.int16)))

            # Normalize SAD by the number of pixels and maximum pixel value
            sad_normalized = sad / (overlap_height * width * 255)

            if sad_normalized < best_score:
                best_score = sad_normalized
                best_shift = shift

            match_percentage = 1 - sad_normalized
            shifts.append(shift)
            match_percentages.append(match_percentage)

            # Visualization every 20 shifts
            if Config["DEBUG_MODE"] and abs(shift) % 100 == 0:
                overlap_size_percentage = overlap_height / height_new  # Normalize overlap size

                plt.suptitle(f"Shift: {shift}\n"
                             f"Overlap Size Percentage: {overlap_size_percentage:.2%}\n"
                             f"Match Percentage: {match_percentage:.2%}")

                # Logging for debugging
                logger.debug(f"Shift: {shift}. Overlapping height: {overlap_height}, "
                             f"Match Percentage: {match_percentage:.2%}")

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

                # Plot match percentage against shift
                axs[2].plot(shifts, match_percentages, label='Match %', color='blue')
                axs[2].set_title("Match Percentage vs. Shift")
                axs[2].set_xlabel("Shift")
                axs[2].set_ylabel("Match Percentage")
                axs[2].legend(loc='upper right')

                plt.draw()
                plt.pause(0.00001) # Adjust pause duration as needed

        plt.close()

        match_score = 1 - best_score
        if match_score >= threshold:
            return best_shift, match_score
        return None, match_score


    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.5):
        # Convert images to grayscale
        base_array_gray = np.array(base_img.convert('L'))
        new_array_gray = np.array(new_img.convert('L'))

        # Ensure images have the same width
        if base_array_gray.shape[1] != new_array_gray.shape[1]:
            raise ValueError("Images must have the same width")

        shift, match_score = ImageMerger.find_image_overlap(base_array_gray, new_array_gray, threshold)

        base_array = np.array(base_img)
        new_array = np.array(new_img)

        if shift is not None:
            if shift >= 0:
                overlap_height = min(base_array.shape[0] - shift, new_array.shape[0])
                blended_overlap = ImageMerger.blend_overlap(base_array[shift:shift + overlap_height], new_array[:overlap_height])
                merged_array = np.vstack((base_array[:shift], blended_overlap, new_array[overlap_height:]))
                logger.info(f"Overlap detected at shift {shift}, overlap height {overlap_height}, match score {match_score:.2f}. Merging...")
            else:
                overlap_height = min(new_array.shape[0] + shift, base_array.shape[0])
                blended_overlap = ImageMerger.blend_overlap(new_array[-shift:-shift + overlap_height], base_array[:overlap_height])
                merged_array = np.vstack((new_array[:-shift], blended_overlap, base_array[overlap_height:]))
                logger.info(f"Overlap detected at shift {shift}, overlap height {overlap_height}, match score {match_score:.2f}. Merging...")
            return Image.fromarray(merged_array)
        else:
            # If no overlap is detected, concatenate the images
            logger.warning("No overlap detected above the threshold, returning original base image")
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
            base_img_path = image_files[0]
            base_img = Image.open(base_img_path)
            image_files_objs = [Image.open(img_path) for img_path in image_files]

            # Merge all subsequent images
            for new_img in image_files_objs[1:]:
                base_img = ImageMerger.merge_images_vertically(base_img, new_img, threshold=0.1)

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
