#!/usr/bin/env python3
import argparse
import ctypes
import queue
import sys
import threading
import time

import mss
import numpy as np
import wx
from PIL import Image
from loguru import logger
from pynput import mouse

from clipboard_manager import ClipboardManager
from image_merger import ImageMerger
from listeners import KeyboardListener, MouseScrollListener
from live_preview import LivePreviewFrame
from region_selector import RegionSelector
from utils import Config, setup_logging

if sys.platform == "win32":
    from hotkey_shortcut import setup_hotkey_shortcut

setup_logging("DEBUG", {"function": True, "thread": True})

MAX_CAPTURE_QUEUE = 3
DEBOUNCE_TIME = 0.5
# A capture that matches the last merged one means the page no longer moves: the user hit the bottom and scrolled again
BOTTOM_PIXEL_THRESHOLD = 16
BOTTOM_MAX_CHANGED_RATIO = 0.001
# Auto scroll starts small, then calibrates wheel clicks so each step moves about this fraction of the region,
# leaving a wide overlap for the matcher
AUTO_SCROLL_START_CLICKS = 2
AUTO_SCROLL_MAX_CLICKS = 15
AUTO_SCROLL_TARGET_RATIO = 0.4

IDLE_HINT = "Scroll until the color panel nears the carriage, then pause."

# Global State
full_merged_image = None
capture_running = True
manual_trigger_event = threading.Event()
# (image, overlay) pairs, so an undo restores the region markers that belonged to that image
undo_stack = []
current_overlay = {}
successful_merge_count = 0
merge_attempts = 0
last_merge_height_added = 0  # 0 when the last attempt failed
auto_scroll_active = threading.Event()
candidate_queue = queue.Queue(maxsize=MAX_CAPTURE_QUEUE)
merge_active = threading.Event()
state_lock = threading.Lock()
status_lock = threading.Lock()
status_state = {
    "capture_state": "idle",
    "merge_state": "idle",
    "merge_progress": None,
    "merge_phase": "",
    "next_capture_in": None,
    "last_result": "",
    "last_result_failed": False,
    "failure_hint": "",
    "recommendation": "",
}


def _capture_screen_region(monitor):
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return img


def run_on_ui_thread(func, *args, **kwargs):
    if wx.IsMainThread():
        return func(*args, **kwargs)

    completed = threading.Event()
    result = {}

    def wrapper():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as exc:
            result["error"] = exc
        finally:
            completed.set()

    wx.CallAfter(wrapper)
    completed.wait()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def publish_status(preview_window=None, **updates):
    global status_state

    with status_lock:
        status_state.update(updates)
        snapshot = dict(status_state)

    with state_lock:
        merged_dimensions = "0x0"
        if full_merged_image is not None:
            merged_dimensions = f"{full_merged_image.width}x{full_merged_image.height}"
        snapshot["total_dimensions"] = merged_dimensions
        snapshot["successful_merges"] = successful_merge_count
        snapshot["can_undo"] = bool(undo_stack) and not merge_active.is_set() and candidate_queue.empty()

    snapshot["queue_size"] = candidate_queue.qsize()
    snapshot["queue_capacity"] = candidate_queue.maxsize
    snapshot["accepting_input"] = not keyboard_listener.exit_event

    if preview_window is not None:
        wx.CallAfter(preview_window.update_status, snapshot)
    elif "preview_window" in globals():
        wx.CallAfter(globals()["preview_window"].update_status, snapshot)


def capture_screenshot(monitor, preview_window=None, settle_delay=0.12):
    preview_was_hidden = False

    if preview_window is not None:
        try:
            if run_on_ui_thread(preview_window.overlaps_region, monitor):
                logger.debug("Preview overlaps capture region; hiding it before screenshot.")
                preview_was_hidden = run_on_ui_thread(preview_window.hide_for_capture)
                if preview_was_hidden:
                    time.sleep(settle_delay)
        except Exception as exc:
            logger.warning(f"Preview overlap guard failed: {exc}")

    try:
        return _capture_screen_region(monitor)
    finally:
        if preview_window is not None and preview_was_hidden:
            try:
                run_on_ui_thread(preview_window.restore_after_capture, preview_was_hidden)
                time.sleep(0.05)
            except Exception as exc:
                logger.warning(f"Preview restore failed: {exc}")


def on_manual_trigger():
    manual_trigger_event.set()
    publish_status(capture_state="manual capture requested", recommendation="Hold steady for the next screenshot")


def on_undo_last():
    global full_merged_image, undo_stack, current_overlay

    if merge_active.is_set() or not candidate_queue.empty():
        logger.info("Undo is unavailable while queued captures are still processing.")
        publish_status(last_result="Undo is unavailable while captures are merging", last_result_failed=False)
        return

    with state_lock:
        if not undo_stack:
            return
        full_merged_image, current_overlay = undo_stack.pop()
        restored_image = full_merged_image.copy()
        restored_overlay = current_overlay

    logger.info(f"Undid last merge. Stack size: {len(undo_stack)}")
    if "preview_window" in globals():
        wx.CallAfter(preview_window.update_image, restored_image, True, restored_overlay)
        wx.CallAfter(preview_window.trace_undo)
    publish_status(last_result="Undid the last successful merge", last_result_failed=False)


def on_cancel():
    if "keyboard_listener" in globals():
        keyboard_listener.request_exit("cancelled")
    publish_status(capture_state="cancelled", recommendation="Closing without copying the stitched image")


def on_done():
    keyboard_listener.request_exit("done")
    publish_status(capture_state="finishing", recommendation="Copying the stitched image once queued captures merge")


def on_auto_scroll(enabled):
    if enabled:
        auto_scroll_active.set()
        publish_status(capture_state="auto scrolling", recommendation="Hands off the mouse; Auto Scroll stops at the bottom")
    else:
        auto_scroll_active.clear()
        publish_status(capture_state="waiting for scroll", recommendation="Auto Scroll stopped; scroll manually or start it again")


def stop_auto_scroll(preview_window, reason):
    auto_scroll_active.clear()
    wx.CallAfter(preview_window.set_auto_scroll, False)
    publish_status(
        preview_window,
        last_result=reason,
        last_result_failed=True,
        failure_hint="Scroll manually or start Auto Scroll again",
    )


def auto_scroll_loop(region, preview_window, keyboard_listener):
    controller = mouse.Controller()
    center = (region["left"] + region["width"] // 2, region["top"] + region["height"] // 2)
    clicks = AUTO_SCROLL_START_CLICKS
    # Wheel clicks scrolled down since the last successful merge, to calibrate and to know how far back to go
    net_clicks = 0

    while not keyboard_listener.exit_event:
        if not auto_scroll_active.wait(timeout=0.1):
            continue
        # Each step is judged on its own merge result, so let queued captures drain first
        if merge_active.is_set() or not candidate_queue.empty():
            time.sleep(0.05)
            continue

        with state_lock:
            attempts_before = merge_attempts
            last_failed = attempts_before > 0 and last_merge_height_added == 0

        # Scroll events go to the window under the pointer
        controller.position = center
        # Never back up all the way: a capture at the last merged position would read as "reached the bottom"
        if last_failed and net_clicks >= 2:
            back = net_clicks // 2
            controller.scroll(0, back)
            net_clicks -= back
            clicks = max(1, clicks // 2)
            logger.info(f"Auto scroll: merge failed, back up {back} clicks")
        elif last_failed:
            stop_auto_scroll(preview_window, "Auto Scroll stopped: captures keep failing even with the smallest scroll")
            continue
        else:
            controller.scroll(0, -clicks)
            net_clicks += clicks

        time.sleep(DEBOUNCE_TIME)
        manual_trigger_event.set()

        while merge_attempts == attempts_before and auto_scroll_active.is_set() and not keyboard_listener.exit_event:
            time.sleep(0.05)

        with state_lock:
            height_added = last_merge_height_added if merge_attempts > attempts_before else 0
        if height_added > 0:
            pixels_per_click = height_added / net_clicks
            clicks = max(1, min(AUTO_SCROLL_MAX_CLICKS, round(region["height"] * AUTO_SCROLL_TARGET_RATIO / pixels_per_click)))
            net_clicks = 0
            logger.info(f"Auto scroll: {pixels_per_click:.0f}px per click, next step {clicks} clicks")


def captures_match(a, b):
    diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max(axis=2)
    return np.count_nonzero(diff > BOTTOM_PIXEL_THRESHOLD) / diff.size < BOTTOM_MAX_CHANGED_RATIO


def merge_worker_loop(preview_window, keyboard_listener):
    global full_merged_image, successful_merge_count, undo_stack, merge_attempts, last_merge_height_added, current_overlay

    last_merged_capture = None

    while True:
        try:
            candidate = candidate_queue.get(timeout=0.05)
        except queue.Empty:
            if keyboard_listener.exit_event:
                break
            continue

        merge_active.set()
        publish_status(
            preview_window,
            merge_state="merging",
            merge_progress=0.0,
            merge_phase="starting",
        )

        with state_lock:
            base_image = full_merged_image.copy() if full_merged_image is not None else None

        if base_image is None:
            candidate_queue.task_done()
            merge_active.clear()
            continue

        if last_merged_capture is not None and captures_match(last_merged_capture, candidate["image"]):
            logger.info("Capture matches the last merged one; reached the bottom. Finishing...")
            keyboard_listener.request_exit("bottom")
            candidate_queue.task_done()
            merge_active.clear()
            continue

        tolerance = 20
        try:
            tolerance = run_on_ui_thread(preview_window.get_tolerance)
        except Exception:
            pass

        merge_started_at = time.time()

        def progress_callback(progress, phase):
            publish_status(
                preview_window,
                merge_state="merging",
                merge_progress=progress,
                merge_phase=phase,
            )

        merged_result, merge_metadata = ImageMerger.merge_images_vertically(
            base_image,
            candidate["image"],
            debug_id=f"live-{candidate['captured_at_ns']}",
            tolerance=tolerance,
            progress_callback=progress_callback,
        )

        if merged_result.height > base_image.height:
            with state_lock:
                old_height = full_merged_image.height if full_merged_image is not None else base_image.height
                full_merged_image = merged_result
                successful_merge_count += 1
                merged_image = full_merged_image.copy()
                height_added = merged_image.height - old_height
                merge_attempts += 1
                last_merge_height_added = height_added
            last_merged_capture = candidate["image"]

            logger.success(
                f"Merged! Total height: {merged_image.height}px "
                f"(Static: top={merge_metadata['static_top']}px, bottom={merge_metadata['static_bottom']}px)"
            )

            debug_info = None
            if Config["DEBUG_MODE"]:
                debug_info = {
                    "total_height": merged_image.height,
                    "height_added": height_added,
                    "processing_time": time.time() - merge_started_at,
                    "debounce_time": max(0.0, candidate["captured_at"] - candidate["scroll_timestamp"]),
                    "static_top": merge_metadata["static_top"],
                    "static_bottom": merge_metadata["static_bottom"],
                    "match_status": merge_metadata["match_status"],
                    "matched_region_start": merge_metadata["matched_region_start"],
                    "matched_region_end": merge_metadata["matched_region_end"],
                    "latest_slice_start": merge_metadata["latest_slice_start"],
                    "latest_slice_end": merge_metadata["latest_slice_end"],
                    "overlap_visual_start": merge_metadata["overlap_visual_start"],
                    "overlap_visual_end": merge_metadata["overlap_visual_end"],
                }
            else:
                debug_info = {
                    "height_added": height_added,
                    "static_top": merge_metadata["static_top"],
                    "static_bottom": merge_metadata["static_bottom"],
                    "match_status": merge_metadata["match_status"],
                    "matched_region_start": merge_metadata["matched_region_start"],
                    "matched_region_end": merge_metadata["matched_region_end"],
                    "latest_slice_start": merge_metadata["latest_slice_start"],
                    "latest_slice_end": merge_metadata["latest_slice_end"],
                    "overlap_visual_start": merge_metadata["overlap_visual_start"],
                    "overlap_visual_end": merge_metadata["overlap_visual_end"],
                }

            with state_lock:
                undo_stack.append((base_image, current_overlay))
                if len(undo_stack) > 10:
                    undo_stack.pop(0)
                current_overlay = debug_info

            wx.CallAfter(preview_window.update_image, merged_image, True, debug_info)
            wx.CallAfter(preview_window.trace_merge, merge_metadata["overlap_visual_start"], merge_metadata["match_score"])
            publish_status(
                preview_window,
                merge_state="idle",
                merge_progress=1.0,
                merge_phase="done",
                last_result=f"Merged +{height_added}px",
                last_result_failed=False,
            )
        else:
            logger.warning("Merge failed (No overlap).")
            with state_lock:
                current_image = full_merged_image.copy() if full_merged_image is not None else base_image.copy()
                merge_attempts += 1
                last_merge_height_added = 0

            failure_debug = None
            failure_last_result = "No overlap found"
            failure_recommendation = "Scroll back slightly so the next capture overlaps the color panel"

            if merge_metadata.get("matched_region_start") is not None:
                failure_debug = {
                    "static_top": merge_metadata["static_top"],
                    "static_bottom": merge_metadata["static_bottom"],
                    "match_status": merge_metadata["match_status"],
                    "matched_region_start": merge_metadata["matched_region_start"],
                    "matched_region_end": merge_metadata["matched_region_end"],
                    "overlap_visual_start": merge_metadata["overlap_visual_start"],
                    "overlap_visual_end": merge_metadata["overlap_visual_end"],
                }
                if Config["DEBUG_MODE"]:
                    failure_debug.update({
                        "total_height": current_image.height,
                        "height_added": 0,
                        "processing_time": time.time() - merge_started_at,
                        "debounce_time": max(0.0, candidate["captured_at"] - candidate["scroll_timestamp"]),
                    })

                failure_last_result = "Nothing new to add"
                failure_recommendation = "Scroll farther, until the color panel nears the carriage"

            wx.CallAfter(preview_window.update_image, current_image, False, failure_debug)
            # A miss with no match at all is logged where the stitch currently ends
            failure_y = merge_metadata["overlap_visual_start"] if merge_metadata["matched_region_start"] is not None else current_image.height
            wx.CallAfter(preview_window.trace_failure, failure_y, merge_metadata["match_score"])
            publish_status(
                preview_window,
                merge_state="idle",
                merge_progress=1.0,
                merge_phase="failed",
                last_result=failure_last_result,
                last_result_failed=True,
                failure_hint=failure_recommendation,
            )

        candidate_queue.task_done()
        merge_active.clear()

    publish_status(preview_window, merge_state="idle", merge_progress=None, merge_phase="")


def processing_loop(region, preview_window, mouse_listener, keyboard_listener):
    global full_merged_image, capture_running, undo_stack, successful_merge_count, current_overlay

    logger.info("Step 1: Capturing initial base image...")
    publish_status(
        preview_window,
        capture_state="capturing initial screenshot",
        merge_state="idle",
        merge_progress=None,
        merge_phase="",
        last_result="Preparing the first screenshot",
        recommendation="Wait for the preview to initialize",
    )

    if keyboard_listener.exit_event:
        wx.CallAfter(wx.GetApp().ExitMainLoop)
        return

    merge_thread = threading.Thread(
        target=merge_worker_loop,
        args=(preview_window, keyboard_listener),
        daemon=True,
    )
    merge_thread.start()

    base_img = capture_screenshot(region, preview_window)
    with state_lock:
        full_merged_image = base_img
        undo_stack.clear()
        current_overlay = {}
        successful_merge_count = 0

    wx.CallAfter(preview_window.update_image, base_img, True)
    publish_status(
        preview_window,
        capture_state="waiting for scroll",
        merge_state="idle",
        merge_progress=None,
        merge_phase="",
        last_result="Base screenshot captured",
        recommendation=IDLE_HINT,
    )

    last_seen_scroll = mouse_listener.last_scroll_time
    pending_scroll_time = None
    last_status_publish = 0.0

    while capture_running and not keyboard_listener.exit_event:
        now = time.time()
        last_scroll = mouse_listener.last_scroll_time
        manual_triggered = manual_trigger_event.is_set()
        scroll_settled = False
        next_capture_in = None

        if last_scroll > last_seen_scroll:
            last_seen_scroll = last_scroll
            pending_scroll_time = last_scroll

        if pending_scroll_time is not None:
            time_since_scroll = now - pending_scroll_time
            if time_since_scroll < DEBOUNCE_TIME:
                next_capture_in = DEBOUNCE_TIME - time_since_scroll
            else:
                scroll_settled = True
                next_capture_in = 0.0

        scroll_trigger_enabled = True
        try:
            scroll_trigger_enabled = preview_window.get_scroll_trigger_enabled()
        except Exception:
            pass
        # Auto scroll triggers its own captures once the page settles; its wheel events must not queue extra ones
        scroll_trigger_enabled = scroll_trigger_enabled and not auto_scroll_active.is_set()

        queue_full = candidate_queue.full()
        if (scroll_settled and scroll_trigger_enabled) or manual_triggered:
            if manual_triggered:
                manual_trigger_event.clear()

            if queue_full:
                publish_status(
                    preview_window,
                    capture_state="queue full",
                    next_capture_in=next_capture_in,
                    last_result="Skipped a capture because the queue is full",
                    recommendation="Pause scrolling until queued captures have merged",
                )
            else:
                publish_status(
                    preview_window,
                    capture_state="capturing screenshot",
                    next_capture_in=None,
                    last_result="Taking a screenshot for the merge queue",
                    recommendation="Hold still until the capture finishes",
                )

                captured_at = time.time()
                new_candidate = capture_screenshot(region, preview_window)
                capture_scroll_time = pending_scroll_time if scroll_settled and pending_scroll_time is not None else captured_at
                candidate_queue.put_nowait({
                    "image": new_candidate,
                    "captured_at": captured_at,
                    "captured_at_ns": time.time_ns(),
                    "scroll_timestamp": capture_scroll_time,
                })
                if scroll_settled:
                    pending_scroll_time = None
                publish_status(
                    preview_window,
                    capture_state="candidate queued",
                    next_capture_in=None,
                    last_result=f"Queued capture {candidate_queue.qsize()}/{candidate_queue.maxsize}",
                    recommendation="Keep scrolling if more content remains; the merge worker is draining the queue",
                )

        # This loop is the only writer of capture_state and recommendation while capturing; results and
        # their failure advice live in their own fields so they survive these refreshes
        if now - last_status_publish >= 0.1:
            if auto_scroll_active.is_set():
                capture_state = "auto scrolling"
                recommendation = "Hands off the mouse; Auto Scroll stops at the bottom"
            elif queue_full:
                capture_state = "queue full"
                recommendation = "Pause scrolling until the merge queue drains"
            elif manual_triggered:
                capture_state = "manual capture requested"
                recommendation = "Hold steady for the manual capture"
            elif pending_scroll_time is None:
                capture_state = "waiting for scroll" if scroll_trigger_enabled else "auto capture disabled"
                recommendation = IDLE_HINT if scroll_trigger_enabled else "Use Take Screenshot or re-enable auto capture"
            elif scroll_settled:
                capture_state = "ready to capture"
                recommendation = "Hold steady; the next screenshot can be taken now"
            else:
                capture_state = "waiting for settle"
                recommendation = "Pause briefly so the next screenshot overlaps cleanly"

            publish_status(
                preview_window,
                capture_state=capture_state,
                next_capture_in=next_capture_in,
                recommendation=recommendation,
            )
            last_status_publish = now

        time.sleep(0.01)

    if not candidate_queue.empty() or merge_active.is_set():
        publish_status(
            preview_window,
            capture_state="draining queue",
            next_capture_in=None,
            recommendation="Waiting for queued captures to finish merging before exit",
        )

    candidate_queue.join()
    merge_thread.join(timeout=1.0)

    should_copy = (
        full_merged_image is not None and
        successful_merge_count > 0 and
        keyboard_listener.exit_reason != "cancelled"
    )

    if should_copy:
        logger.info("Copying to clipboard...")
        with state_lock:
            final_image = full_merged_image.copy()
        wx.CallAfter(preview_window.update_image, final_image, True)
        publish_status(
            preview_window,
            capture_state="completed",
            merge_state="idle",
            merge_progress=None,
            merge_phase="",
            last_result="Copied the stitched image to the clipboard",
            last_result_failed=False,
            recommendation="Done",
        )
        ClipboardManager.copy_image_to_clipboard(final_image)

        if Config["DEBUG_MODE"]:
            final_image.show()
    else:
        logger.info("Exiting without copying stitched output.")
        publish_status(
            preview_window,
            capture_state="cancelled" if keyboard_listener.exit_reason == "cancelled" else "stopped",
            merge_state="idle",
            merge_progress=None,
            merge_phase="",
            last_result="Exited without copying stitched output",
            last_result_failed=False,
            recommendation="Done",
        )

    wx.CallAfter(wx.GetApp().ExitMainLoop)


def main():
    global capture_running

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    Config["DEBUG_MODE"] = args.debug

    if sys.platform == "win32":
        # mss makes the process per-monitor DPI aware on its first grab. Do it before any wx window
        # exists so the selection coordinates and the captures are both in physical pixels.
        ctypes.oledll.shcore.SetProcessDpiAwareness(2)
        # Only the packaged exe gets a hotkey; from source there is no single file for a shortcut to launch
        if getattr(sys, "frozen", False) and not setup_hotkey_shortcut():
            return

    logger.info("Select region...")
    selector = RegionSelector()
    selection = selector.select_region()
    if not selection:
        return

    key_listener = KeyboardListener()
    key_listener.start()
    globals()["keyboard_listener"] = key_listener

    mouse_listener = MouseScrollListener(key_listener)
    mouse_listener.start()

    app = wx.App(False)
    # Before the preview: both stay on top, and the one shown last ends up above
    veil = selector.highlight_region(selection)
    preview = LivePreviewFrame(
        selection["height"],
        debug_mode=Config["DEBUG_MODE"],
        selection_region=selection,
        manual_callback=on_manual_trigger,
        undo_callback=on_undo_last,
        cancel_callback=on_cancel,
        done_callback=on_done,
        veil_callback=veil.Show,
        auto_scroll_callback=on_auto_scroll,
    )
    globals()["preview_window"] = preview
    threading.Thread(target=auto_scroll_loop, args=(selection, preview, key_listener), daemon=True).start()

    worker_thread = threading.Thread(
        target=processing_loop,
        args=(selection, preview, mouse_listener, key_listener),
        daemon=True,
    )
    wx.CallAfter(worker_thread.start)

    logger.info("System Ready.")
    logger.info("1. Scroll the content.")
    logger.info("2. Pause to queue a capture.")
    logger.info("3. Watch the preview status for queue and merge progress.")

    app.MainLoop()


if __name__ == "__main__":
    main()
