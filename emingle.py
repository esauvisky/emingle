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
    screenshots = [capture_screenshot(selection)]
    logger.info("Captured initial screenshot.")

    while not keyboard_listener.exit_event:
        if mouse_listener.screenshot_event:
            mouse_listener.screenshot_event = False
            img = capture_screenshot(selection)
            logger.info("Captured image.")
            screenshots.append(img)
        time.sleep(0.01) # Prevent busy waiting

    logger.info("Merging screenshots...")

    # Start with the first image as the base

    # Find and remove fixed borders
    top, bottom, left, right = ImageMerger.find_fixed_borders(screenshots)
    logger.info(f"Cropping images to region: top={top}, bottom={bottom}, left={left}, right={right}")

    cropped_images = []
    for img in screenshots:
        img_array = np.array(img)
        cropped_array = ImageMerger.remove_borders(img_array, top, bottom, left, right)
        cropped_img = Image.fromarray(cropped_array)
        cropped_images.append(cropped_img)

    logger.info(f"Images dimensions: {[img.size for img in screenshots]}")
    merged_image = cropped_images[0]
    for new_img in cropped_images[1:]:
        merged_image = ImageMerger.merge_images_vertically(
            merged_image, new_img, threshold=0.5
        )
        if merged_image is None:
            logger.warning(f"Failed to merge screenshot. Skipping.")

    if merged_image is not None:
        if Config["DEBUG_MODE"]:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Saving debug images to: {temp_dir}")
            for i, screenshot in enumerate(screenshots):
                screenshot.save(os.path.join(temp_dir, f"screenshot_{i}.png"))
            merged_image.save(os.path.join(temp_dir, "merged_screenshot.png"))
            merged_image.show()

        logger.info("Copying merged image to clipboard...")
        # app.ExitMainLoop()
        ClipboardManager.copy_image_to_clipboard(merged_image)
    else:
        logger.error("No screenshots were taken, nothing to copy to clipboard.")


if __name__ == "__main__":
    main()
