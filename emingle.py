#!/usr/bin/env python3
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from utils import setup_logging, Config
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA is not available. Using CPU.")

setup_logging("DEBUG", {"function": True, "thread": True})
import os
# os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from image_merger import ImageMerger
from clipboard_manager import ClipboardManager
from listeners import KeyboardListener, MouseScrollListener, WxAppRunner
from region_selector import RegionSelector

import mss
from PIL import Image
import argparse
import sys
import time
import queue
import threading
import os
import tempfile

merged_image = None
highlight_app = None


def capture_screenshot(monitor):
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img

def main():
    global merged_image
    parser = argparse.ArgumentParser(description='Screenshot merger.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to display images during merging.')
    args = parser.parse_args()
    Config["DEBUG_MODE"] = args.debug

    logger.info("Select the region to capture.")
    region_selector = RegionSelector()
    selection = region_selector.select_region()
    if not selection:
        logger.error("No region selected. Exiting.")
        sys.exit(0)

    logger.info(f"Selected region: {selection}")

    if not Config["DEBUG_MODE"]:
        app = region_selector.highlight_region(selection["left"], selection["top"], selection["width"], selection["height"])
        wx_app_runner = WxAppRunner(app)
        wx_app_runner.start()

    keyboard_listener = KeyboardListener()
    keyboard_listener.start()

    mouse_listener = MouseScrollListener(keyboard_listener)
    mouse_listener.start()

    logger.info("\nInstructions:\n"
                " - The script will capture a screenshot after every 5 mouse scrolls.\n"
                " - Scroll the underlying content to capture new screenshots.\n"
                " - Press Escape to finish capturing, merge images and send to your clipboard.\n")

    # Take initial screenshot
    initial_screenshot = capture_screenshot(selection)
    screenshots = [initial_screenshot]
    logger.info("Captured initial screenshot.")

    while not keyboard_listener.exit_event:
        if mouse_listener.screenshot_event:
            mouse_listener.screenshot_event = False
            img = capture_screenshot(selection)
            logger.info("Captured image.")
            screenshots.append(img)
        time.sleep(0.01) # Prevent busy waiting

    logger.info("Merging screenshots...")

    # Find and identify fixed top and bottom borders
    top, bottom, left_ignore, right_ignore = ImageMerger.find_fixed_borders(screenshots)
    # The left and right values from find_fixed_borders are ignored as we want to keep the full width

    logger.info(f"Cropping images vertically: top={top}, bottom={bottom}")

    processed_images = []
    for img in screenshots:
        img_array = np.array(img)
        # remove_borders now only crops vertically, retaining full width
        processed_array = ImageMerger.remove_borders(img_array, top, bottom, 0, img_array.shape[1])
        processed_img = Image.fromarray(processed_array)
        processed_images.append(processed_img)

    logger.info(f"Images dimensions (after vertical cropping): {[img.size for img in processed_images]}")
    merged_image = processed_images[0]
    for new_img in processed_images[1:]:
        # Merge the vertically-cropped, full-width images
        merged_image = ImageMerger.merge_images_vertically(
            merged_image, new_img, threshold=0.5
        )
        if merged_image is None:
            logger.warning(f"Failed to merge screenshot. The process will continue with the last successful merge.")
            # If a merge fails (e.g., no overlap found), merged_image will remain the previous base image.
            # This allows subsequent images to try and merge with it.

    if merged_image is not None:
        # No need to add borders back, as horizontal borders were never removed.
        final_image_to_clipboard = merged_image

        if Config["DEBUG_MODE"]:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Saving debug images to: {temp_dir}")
            for i, screenshot in enumerate(screenshots):
                screenshot.save(os.path.join(temp_dir, f"screenshot_{i}.png"))
            final_image_to_clipboard.save(os.path.join(temp_dir, "merged_screenshot.png"))
            final_image_to_clipboard.show()

        logger.info("Copying merged image to clipboard...")
        ClipboardManager.copy_image_to_clipboard(final_image_to_clipboard)
    else:
        logger.error("No screenshots were captured or merged successfully, nothing to copy to clipboard.")


if __name__ == "__main__":
    main()
