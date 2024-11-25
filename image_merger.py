import filecmp
import glob
from math import sqrt
import os
import random
import time

from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from utils import Config


class ImageMerger:
    @staticmethod
    def find_fixed_borders(images, margin=0.2):
        rgb_arrays = [np.array(img) for img in images]

        # # Display grid of images
        # grid_size = int(sqrt(len(images)))
        # fig, axs = plt.subplots(2, len(images)//2, figsize=(25, 10))
        # for i, img in enumerate(images[:len(images)//2*2]):
        #     axs[i%2][i//2].clear()
        #     axs[i%2][i//2].imshow(img)
        #     axs[i%2][i//2].axis('off')
        # plt.show()

        # input("Press Enter to continue...")

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
    def find_image_overlap(base_array_gray, new_array_gray, threshold=0.5, z_score_threshold=2.0):
        height_base, width = base_array_gray.shape
        height_new, _ = new_array_gray.shape

        best_shift = None
        best_score = np.inf  # Initialize with a large value
        best_zscore = 0

        fig = None
        axs = []
        if Config["DEBUG_MODE"]:
            fig, axs = plt.subplots(1, 3, figsize=(10, 8))  # Create three subplots side by side

        shifts = []
        best_scores = []
        zscores = []

        # Define the range for early shifts and delayed shifts
        margin = height_new // 10
        first_list = list(range(-height_new + 1 + margin, height_base // 2))
        second_list = list(range(height_base - margin, height_base // 2, -1))
        offsets = []
        for x, y in zip(first_list, second_list):
            offsets.append(x)
            offsets.append(y)
        # offsets.extend(list(range(0, height_base - height_new + 1)))

        # Combine shifts prioritizing early and delayed shifts, then check remaining
        for shift in offsets:
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
            sad = np.sum(np.abs(base_overlap - new_overlap))

            # Normalize SAD by the number of pixels and maximum pixel value
            sad_normalized = sad / (overlap_height * width * 255)

            match_percentage = 1 - sad_normalized
            shifts.append(shift)

            # Calculate mean and standard deviation of match percentages
            best_scores.append(match_percentage)
            mean_match = np.mean(best_scores)
            std_match = np.std(best_scores)

            # Determine if there is an outlier (a spike in match percentage)
            z_score = abs((match_percentage - mean_match) / std_match) if std_match > 0 else 0
            zscores.append(z_score)

            if sad_normalized < best_score and z_score > 2:
                best_zscore = z_score
                best_score = sad_normalized
                best_shift = shift

                # if z_score > z_score_threshold:
                #     logger.info(f"Spike detected: Match percentage at shift {shift} is {match_percentage:.8%} with Z-score {z_score:.2f}. Returning early.")
                #     return shift, match_percentage
                # else:
                #     logger.debug(f"Match percentage at shift {shift} is {match_percentage:.8%} with Z-score {z_score:.2f}.")
            # else:
            #     zscores.append(0)

            # Visualization every 20 shifts
            if Config["DEBUG_MODE"] and abs(shift) % 200 == 0:
                overlap_size_percentage = overlap_height / height_new  # Normalize overlap size

                plt.suptitle(f"Shift: {shift}\n"
                             f"Overlap Size Percentage: {overlap_size_percentage:.0%}\n"
                             f"Match Percentage: {match_percentage:.8%}\n"
                             f"Z-score: {z_score:.2f}")

                # Logging for debugging
                logger.debug(f"Shift: {shift}. Overlapping height: {overlap_height}, "
                             f"Match Percentage: {match_percentage:.8%}. Z-score: {z_score:.2f}.")

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
                # axs[2].plot(shifts, match_percentages, label='Match %', color='blue')
                axs[2].set_title("Match Percentage vs. Shift")
                axs[2].set_xlabel("Shift")
                axs[2].set_ylabel("Match Percentage")
                axs[2].scatter(shifts, best_scores, color='red')
                # plot zscores
                #  set small size and separate y scale
                # axs[2].scatter(shifts, zscores, color='green')
                # # axs[2].set_yscale('log')
                # axs[2].set_ylim(0, 1)
                # axs[2].set_xlim(-height_new + 1, height_base - height_new)
                # axs[2].set_title("Match Percentage vs. Shift")
                # axs[2].set_xlabel("Shift")
                # axs[2].set_ylabel("Match Percentage")

                plt.draw()
                plt.pause(0.00001)  # Adjust pause duration as needed

        plt.close()

        match_score = 1 - best_score
        if match_score >= threshold:
            return best_shift, match_score, best_zscore
        return None, match_score, best_zscore


    @staticmethod
    def merge_images_vertically(base_img, new_img, threshold=0.5):
        # Convert images to grayscale
        base_array_gray = np.array(base_img.convert('L'))
        new_array_gray = np.array(new_img.convert('L'))

        # Ensure images have the same width
        if base_array_gray.shape[1] != new_array_gray.shape[1]:
            raise ValueError(f"Images must have the same width. Base width: {base_array_gray.shape[1]}, New width: {new_array_gray.shape[1]}")

        shift, match_score, zscore = ImageMerger.find_image_overlap(base_array_gray, new_array_gray, threshold)

        base_array = np.array(base_img)
        new_array = np.array(new_img)

        def visualize(base_img):
            # Visualize the merged image in real-time
            plt.imshow(base_img)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)  # Adjust pause duration as needed
            plt.clf()  # Clear the figure for the next update

        if shift is not None:
            if shift >= 0:
                overlap_height = min(base_array.shape[0] - shift, new_array.shape[0])
                blended_overlap = ImageMerger.blend_overlap(base_array[shift:shift + overlap_height], new_array[:overlap_height])
                merged_array = np.vstack((base_array[:shift], blended_overlap, new_array[overlap_height:]))
                logger.info(f"Overlap detected at shift {shift}, overlap height {overlap_height}, match score {match_score:.8f}, zscore {zscore:.2f}. Merging...")
            else:
                overlap_height = min(new_array.shape[0] + shift, base_array.shape[0])
                blended_overlap = ImageMerger.blend_overlap(new_array[-shift:-shift + overlap_height], base_array[:overlap_height])
                merged_array = np.vstack((new_array[:-shift], blended_overlap, base_array[overlap_height:]))
                logger.info(f"Overlap detected at shift {shift}, overlap height {overlap_height}, match score {match_score:.8f}, zscore {zscore:.2f}. Merging...")

            if Config["DEBUG_MODE"]:
                visualize(Image.fromarray(merged_array))
            return Image.fromarray(merged_array)
        else:
            # If no overlap is detected, concatenate the images
            logger.warning("No overlap detected above the threshold, returning original base image")
            if Config["DEBUG_MODE"]:
                visualize(Image.fromarray(base_img))
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
                base_img = ImageMerger.merge_images_vertically(base_img, new_img, threshold=0.5)

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
