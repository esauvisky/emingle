#!/usr/bin/env python3
from loguru import logger
import numpy as np
from utils import setup_logging, Config

setup_logging("DEBUG", {"function": True, "thread": True})

from image_merger import ImageMerger
from clipboard_manager import ClipboardManager
from listeners import KeyboardListener, MouseScrollListener
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

    time.sleep(0.1)
    logger.info(f"Selected region: {selection}")
    keyboard_listener = KeyboardListener()
    keyboard_listener.start()

    mouse_listener = MouseScrollListener()
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
        time.sleep(0.001) # Prevent busy waiting

    logger.info("Merging screenshots...")

    # Start with the first image as the base
    # top, bottom, left, right = ImageMerger.find_fixed_borders(screenshots)
    # logger.info(f"Images dimensions: {[img.size for img in screenshots]}")
    # logger.info(f"Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}")
    # screenshots = [ImageMerger.remove_borders(np.array(img), top, bottom, left, right) for img in screenshots]
    # screenshots = [Image.fromarray(img) for img in screenshots]
    merged_image = screenshots[0] if screenshots else None
    logger.info(f"Images dimensions: {[img.size for img in screenshots]}")
    for i in range(1, len(screenshots)):
        merged_image = ImageMerger.merge_images_vertically(merged_image, screenshots[i])
        if merged_image is None:
            logger.warning(f"Failed to merge screenshot {i+1}. Skipping.")

    if merged_image is not None:
        if Config["DEBUG_MODE"]:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Saving debug images to: {temp_dir}")
            for i, screenshot in enumerate(screenshots):
                screenshot.save(os.path.join(temp_dir, f"screenshot_{i}.png"))
            merged_image.save(os.path.join(temp_dir, "merged_screenshot.png"))

        logger.info("Copying merged image to clipboard...")
        ClipboardManager.copy_image_to_clipboard(merged_image)
    else:
        logger.error("No screenshots were taken, nothing to copy to clipboard.")


if __name__ == "__main__":
    main()
