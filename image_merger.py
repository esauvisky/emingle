import filecmp
import glob
import math
import os
import random
import time

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from loguru import logger


class ImageMerger:
    @staticmethod


    def find_image_overlap(base_array, new_array, threshold=0.05):
        height_base, width_base = base_array.shape
        height_new, width_new = new_array.shape

        # Create shifts from the negative height of new_array to the positive height of base_array
        shifts = np.arange(-height_new + 1, height_base)  # Range from (-height_new + 1) to (height_base - 1)

        match_percentages = np.zeros(shifts.shape[0], dtype=np.float32)


        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create two subplots side by side
        for ix, shift in enumerate(shifts):
            if shift > 0 and shift < height_base - height_new:
                continue

            # Calculate the overlap ranges
            if shift < 0:
                # Negative shift: new_array extends above the top of base_array
                start_base = 0
                end_base = height_new + shift  # Overlapping part in base_array
                start_new = -shift  # Starting point in new_array to match base_array
                end_new = height_new  # Entire height of new_array
            else:
                # Positive shift: new_array starts lower in base_array
                start_base = shift
                end_base = min(height_base, shift + height_new)  # Ensure we don't exceed base_array
                start_new = 0  # Starting from the beginning of new_array
                end_new = end_base - start_base  # Overlapping height

            # Extract the overlapping regions
            base_overlap = base_array[start_base:end_base]
            new_overlap = new_array[start_new:end_new]

            # Calculate matching percentage

            match_percentages[ix] = np.mean(base_overlap == new_overlap) * (1 - (end_base - start_base) / height_new)

            plt.suptitle(f"Shift: {shift}\nOverlap Size Percentage: {((end_base - start_base) / height_new):.2%}\nMatch Percentage: {match_percentages[ix]:.2%}")
            # Visualization every 10 shifts
            if abs(shift) % 10 == 0:
                # plt.clf()  # Clear the figure for the next update
                # plt.ion()  # Turn on interactive mode for real-time plotting
                # Update the subplots to show base_overlap and new_overlap
                axs[0].clear()
                axs[1].clear()

                # Display the overlapping regions side by side
                axs[0].imshow(base_overlap, cmap='gray')
                axs[0].set_title(f"Base Overlap\nStart: {start_base}, End: {end_base}")
                axs[0].axis('off')

                axs[1].imshow(new_overlap, cmap='gray')
                axs[1].set_title(f"New Overlap\nStart: {start_new}, End: {end_new}")
                axs[1].axis('off')

                plt.draw()
                # Draw the updates
                plt.pause(0.001)  # Adjust pause duration as needed
                # plt.pause(0.001)  # Adjust pause duration as needed

        # plt.ioff()  # Turn off interactive mode
        # plt.show()  # Show the last plot
        plt.close()

        # Find the best match shift and its corresponding percentage
        best_match_index = np.argmax(match_percentages)
        best_match_percentage = match_percentages[best_match_index]
        best_shift = shifts[best_match_index] if best_match_percentage >= threshold else None

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
            overlap_size = min(height_new, height_base - shift)  # Ensure we don't exceed base_array

            if shift < 0:
                merged_array = np.vstack((new_array[0:-shift], base_array))
                logger.info(f"Overlap detected at height {shift} with {overlap_size} rows of overlap and match percentage {match_percentage:.2%}. Merging...")
            else:
                merged_array = np.vstack((base_array[0:shift], new_array))
                logger.info(f"Overlap detected at height {shift} with {overlap_size} rows of overlap and match percentage {match_percentage:.2%}. Merging...")

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

    tests = sorted(os.listdir(base_dir))
    random.shuffle(tests)
    for subdir in tests:
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
                    # input()
            else:
                logger.warning(
                    f"Existing merged image not found at {existing_merged_path}. Saving merged image."
                )

            # Clean up the temporary merged image
            if os.path.exists(merged_temp_path):
                os.remove(merged_temp_path)

            logger.info(f"Completed processing directory: {subdir_path} in {time.time() - dir_start_time:.2f} seconds")

    logger.info("Image processing completed.")
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
