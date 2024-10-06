#!/usr/bin/env python3
from loguru import logger
from utils import setup_logging

setup_logging("DEBUG", {"function": True, "thread": True})

from image_merger import ImageMerger
from clipboard_manager import ClipboardManager
from listeners import KeyboardListener, MouseScrollListener
from region_selector import RegionSelector

import mss
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import sys
import time
import queue
import threading

merged_image = None


def capture_screenshot(monitor):
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img


# Function to display images in debug mode
def display_images(new_image, merged_image):
    root = tk.Tk(sync=False)
    root.title("Debug Images")

    # Create frames for layout
    frame_new = tk.Frame(root)
    frame_new.pack(side="left", padx=10, pady=10)

    frame_merged = tk.Frame(root)
    frame_merged.pack(side="right", padx=10, pady=10)

    # New Captured Image
    new_image_tk = ImageTk.PhotoImage(new_image)
    new_image_label = tk.Label(frame_new, image=new_image_tk) # type: ignore
    new_image_label.pack()
    new_image_title = tk.Label(frame_new, text="New Captured Image")
    new_image_title.pack()

    # Merged Image
    if merged_image is not None:
        merged_image_tk = ImageTk.PhotoImage(merged_image)
        merged_image_label = tk.Label(frame_merged, image=merged_image_tk) # type: ignore
        merged_image_label.pack()
        merged_image_title = tk.Label(frame_merged, text="Current Merged Image")
        merged_image_title.pack()

    # Keep a reference to the images to prevent garbage collection
    root.mainloop()


def merge_images_thread(screenshot_queue, exit_event):
    global merged_image
    while not exit_event.is_set() or not screenshot_queue.empty():
        try:
            img = screenshot_queue.get(timeout=1)
            new_merged = ImageMerger.merge_images_vertically(merged_image, img)
            if new_merged is not None:
                logger.info("Successfully merged with existing image.")
                merged_image = new_merged

            screenshot_queue.task_done()
        except queue.Empty:
            continue


def main():
    global merged_image
    parser = argparse.ArgumentParser(description='Screenshot merger.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to display images during merging.')
    args = parser.parse_args()
    debug_mode = args.debug

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

    merged_image = capture_screenshot(selection)
    screenshot_queue = queue.Queue()
    exit_event = threading.Event()

    merge_thread = threading.Thread(target=merge_images_thread, args=(screenshot_queue, exit_event))
    merge_thread.start()

    while not keyboard_listener.exit_event:
        if mouse_listener.screenshot_event:
            mouse_listener.screenshot_event = False
            img = capture_screenshot(selection)
            logger.info("Captured image.")
            screenshot_queue.put(img)
        time.sleep(0.1) # Prevent busy waiting

    exit_event.set()
    logger.info("Waiting for all screenshots to be merged...")
    screenshot_queue.join()
    merge_thread.join()

    logger.info("Copying merged image to clipboard...")
    ClipboardManager.copy_image_to_clipboard(merged_image)


if __name__ == "__main__":
    main()
