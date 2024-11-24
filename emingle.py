#!/usr/bin/env python3
from loguru import logger
from utils import setup_logging

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

DEBUG_MODE = False

def main():
    global merged_image, DEBUG_MODE
    parser = argparse.ArgumentParser(description='Screenshot merger.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to display images during merging.')
    args = parser.parse_args()
    DEBUG_MODE = args.debug

    logger.info("Select the region to capture.")
    region_selector = RegionSelector()
    selection = region_selector.select_region()
    if not selection:
        logger.error("No region selected. Exiting.")
        sys.exit(0)

    logger.info(f"Selected region: {selection}")

    keyboard_listener = KeyboardListener()
    keyboard_listener.start()

    mouse_listener = MouseScrollListener()
    mouse_listener.start()

    logger.info("\nInstructions:\n"
                " - The script will capture a screenshot after every 5 mouse scrolls.\n"
                " - Scroll the underlying content to capture new screenshots.\n"
                " - Press Escape to finish capturing, merge images and send to your clipboard.\n")

    screenshots = []
    while not keyboard_listener.exit_event:
        if mouse_listener.screenshot_event:
            mouse_listener.screenshot_event = False
            img = capture_screenshot(selection)
            logger.info("Captured image.")
            screenshots.append(img)
        time.sleep(0.1) # Prevent busy waiting

    logger.info("Merging screenshots...")
    merged_image = screenshots[0] if screenshots else None
    for i in range(1, len(screenshots)):
        merged_image = ImageMerger.merge_images_vertically(merged_image, screenshots[i])
        if merged_image is None:
            logger.warning(f"Failed to merge screenshot {i+1}. Skipping.")

    if merged_image is not None:
        if DEBUG_MODE:
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
