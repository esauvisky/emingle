#!/usr/bin/env python3
import sys
import time
import threading
import wx
import numpy as np
import mss
from PIL import Image
import argparse
from loguru import logger

from utils import setup_logging, Config
from image_merger import ImageMerger
from clipboard_manager import ClipboardManager
from listeners import KeyboardListener, MouseScrollListener
from region_selector import RegionSelector
from live_preview import LivePreviewFrame

setup_logging("DEBUG", {"function": True, "thread": True})

# Global State
full_merged_image = None
capture_running = True

def capture_screenshot(monitor):
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img

def processing_loop(region, preview_window, mouse_listener, keyboard_listener):
    global full_merged_image, capture_running

    # Configuration
    DEBOUNCE_TIME = 0.3  # Seconds to wait after scrolling stops

    logger.info("Step 1: Capturing initial base image...")

    # 1. Initial Capture
    base_img = capture_screenshot(region)
    full_merged_image = base_img

    wx.CallAfter(preview_window.update_image, full_merged_image, "Started. Scroll & Stop to capture.", True)

    # Mark the time so we don't re-capture the initial state immediately
    last_processed_time = time.time()

    while capture_running and not keyboard_listener.exit_event:

        now = time.time()
        last_scroll = mouse_listener.last_scroll_time

        # Logic:
        # 1. Has there been a scroll AFTER our last processing?
        # 2. Has enough time passed since that scroll (is it settled)?

        if last_scroll > last_processed_time:
            time_since_scroll = now - last_scroll

            if time_since_scroll < DEBOUNCE_TIME:
                # User is still scrolling or just stopped. Update status but wait.
                # wx.CallAfter(preview_window.status_label.SetLabel, "Settling...")
                pass
            else:
                # --- DEBOUNCE TRIGGERED ---
                logger.info(f"Scroll settled ({time_since_scroll:.2f}s). Capturing...")

                # Update timestamp immediately to prevent double triggers
                # This ensures debounce only counts time between scrolls, not processing time
                last_processed_time = last_scroll

                new_candidate = capture_screenshot(region)

                # Attempt Merge
                merged_result = ImageMerger.merge_images_vertically(
                    full_merged_image,
                    new_candidate,
                    debug_id="live"
                )

                if merged_result.height > full_merged_image.height:
                    # Success
                    old_height = full_merged_image.height
                    full_merged_image = merged_result
                    logger.success(f"Merged! Total height: {full_merged_image.height}px")

                    # Pass debug info if in debug mode
                    debug_info = None
                    if Config["DEBUG_MODE"]:
                        debug_info = {
                            'total_height': full_merged_image.height,
                            'height_added': full_merged_image.height - old_height,
                            'processing_time': time.time() - last_scroll,
                            'debounce_time': time_since_scroll
                        }

                    wx.CallAfter(preview_window.update_image, full_merged_image, "Merged! Keep scrolling.", True, debug_info)
                else:
                    # Fail
                    logger.warning("Merge failed (No overlap).")
                    wx.CallAfter(preview_window.update_image, full_merged_image, "MISMATCH! Scroll UP slightly.", False)

        time.sleep(0.005)

    # --- FINALIZATION ---
    if full_merged_image:
        logger.info("Copying to clipboard...")
        wx.CallAfter(preview_window.update_image, full_merged_image, "Copied to Clipboard!", True)
        ClipboardManager.copy_image_to_clipboard(full_merged_image)

        if Config["DEBUG_MODE"]:
            full_merged_image.show()

    wx.CallAfter(wx.GetApp().ExitMainLoop)

def main():
    global capture_running
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    Config["DEBUG_MODE"] = args.debug

    logger.info("Select region...")
    selector = RegionSelector()
    selection = selector.select_region()
    if not selection: return

    # Start Listeners
    key_listener = KeyboardListener()
    key_listener.start()

    mouse_listener = MouseScrollListener(key_listener)
    mouse_listener.start()

    # UI
    app = wx.App(False)
    preview = LivePreviewFrame(selection['height'], debug_mode=Config["DEBUG_MODE"], selection_region=selection)

    # Thread
    t = threading.Thread(
        target=processing_loop,
        args=(selection, preview, mouse_listener, key_listener),
        daemon=True
    )
    t.start()

    logger.info("System Ready.")
    logger.info("1. Scroll the content.")
    logger.info("2. Stop and wait 0.6s.")
    logger.info("3. Watch the preview window.")

    app.MainLoop()

if __name__ == "__main__":
    main()
