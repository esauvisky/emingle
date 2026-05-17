#!/usr/bin/env python3
import argparse
import queue
import threading
import time

import mss
import wx
from PIL import Image
from loguru import logger

from clipboard_manager import ClipboardManager
from image_merger import ImageMerger
from listeners import KeyboardListener, MouseScrollListener
from live_preview import LivePreviewFrame
from region_selector import RegionSelector
from utils import Config, setup_logging

setup_logging("DEBUG", {"function": True, "thread": True})

MAX_CAPTURE_QUEUE = 3
DEBOUNCE_TIME = 0.5

# Global State
full_merged_image = None
capture_running = True
manual_trigger_event = threading.Event()
undo_stack = []
successful_merge_count = 0
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
    "recommendation": "",
    "hover_hint": None,
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

    snapshot["queue_size"] = candidate_queue.qsize()
    snapshot["queue_capacity"] = candidate_queue.maxsize

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
    global full_merged_image, undo_stack

    if merge_active.is_set() or not candidate_queue.empty():
        logger.info("Undo is unavailable while queued captures are still processing.")
        publish_status(
            last_result="Undo unavailable while queued captures are still merging",
            recommendation="Wait for the queue to drain before undoing",
        )
        return

    with state_lock:
        if not undo_stack:
            return
        full_merged_image = undo_stack.pop()
        restored_image = full_merged_image.copy()

    logger.info(f"Undid last merge. Stack size: {len(undo_stack)}")
    if "preview_window" in globals():
        wx.CallAfter(preview_window.update_image, restored_image, "Undid last merge", True)
    publish_status(last_result="Undid the last successful merge", recommendation="Scroll again when the latest slice is near the bottom")


def on_cancel():
    if "keyboard_listener" in globals():
        keyboard_listener.request_exit("cancelled")
    publish_status(capture_state="cancelled", recommendation="Closing without copying the stitched image")


def on_preview_hover(label):
    publish_status(hover_hint=f"hover: {label}" if label else None)


def merge_worker_loop(preview_window, keyboard_listener):
    global full_merged_image, successful_merge_count, undo_stack

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
            recommendation="You can keep scrolling while this merge finishes",
        )

        with state_lock:
            base_image = full_merged_image.copy() if full_merged_image is not None else None

        if base_image is None:
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
                recommendation="Wait for the merge to finish or keep scrolling if more content remains",
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
                undo_stack.append(base_image.copy())
                if len(undo_stack) > 10:
                    undo_stack.pop(0)

                old_height = full_merged_image.height if full_merged_image is not None else base_image.height
                full_merged_image = merged_result
                successful_merge_count += 1
                merged_image = full_merged_image.copy()
                height_added = merged_image.height - old_height

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

            wx.CallAfter(preview_window.update_image, merged_image, "Merged! Keep scrolling.", True, debug_info)
            publish_status(
                preview_window,
                merge_state="idle",
                merge_progress=1.0,
                merge_phase="done",
                last_result=f"Merged +{height_added}px from a queued capture",
                recommendation="Scroll until the color slice is almost out of view before capturing again",
            )
        else:
            logger.warning("Merge failed (No overlap).")
            with state_lock:
                current_image = full_merged_image.copy() if full_merged_image is not None else base_image.copy()

            failure_debug = None
            failure_last_result = "Merge failed; the new screenshot did not overlap enough"
            failure_recommendation = "Scroll back slightly and let the next capture include more overlap"

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

                failure_last_result = "Capture matched earlier content but did not extend the stitch"
                failure_recommendation = "Scroll farther so the colored slice moves closer to the bottom before capturing again"

            wx.CallAfter(preview_window.update_image, current_image, "MISMATCH! Scroll UP slightly.", False, failure_debug)
            publish_status(
                preview_window,
                merge_state="idle",
                merge_progress=1.0,
                merge_phase="failed",
                last_result=failure_last_result,
                recommendation=failure_recommendation,
            )

        candidate_queue.task_done()
        merge_active.clear()

    publish_status(preview_window, merge_state="idle", merge_progress=None, merge_phase="")


def processing_loop(region, preview_window, mouse_listener, keyboard_listener):
    global full_merged_image, capture_running, undo_stack, successful_merge_count

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
        successful_merge_count = 0

    wx.CallAfter(preview_window.update_image, base_img, "Started. Scroll & Stop to capture.", True)
    publish_status(
        preview_window,
        capture_state="waiting for scroll",
        merge_state="idle",
        merge_progress=None,
        merge_phase="",
        last_result="Base screenshot captured",
        recommendation="Scroll until the latest visible content is close to the bottom, then pause briefly",
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

        if now - last_status_publish >= 0.1:
            if queue_full:
                capture_state = "queue full"
                recommendation = "Pause scrolling until the merge queue drains"
            elif manual_triggered:
                capture_state = "manual capture requested"
                recommendation = "Hold steady for the manual capture"
            elif pending_scroll_time is None:
                capture_state = "waiting for scroll" if scroll_trigger_enabled else "auto capture disabled"
                recommendation = (
                    "Scroll down until the color slice is almost out of view"
                    if scroll_trigger_enabled else
                    "Use Take Screenshot or re-enable auto capture"
                )
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
        wx.CallAfter(preview_window.update_image, final_image, "Copied to Clipboard!", True)
        publish_status(
            preview_window,
            capture_state="completed",
            merge_state="idle",
            merge_progress=None,
            merge_phase="",
            last_result="Copied the stitched image to the clipboard",
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
            recommendation="Done",
        )

    wx.CallAfter(wx.GetApp().ExitMainLoop)


def main():
    global capture_running

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    Config["DEBUG_MODE"] = args.debug

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
    preview = LivePreviewFrame(
        selection["height"],
        debug_mode=Config["DEBUG_MODE"],
        selection_region=selection,
        manual_callback=on_manual_trigger,
        undo_callback=on_undo_last,
        cancel_callback=on_cancel,
        hover_callback=on_preview_hover,
    )
    globals()["preview_window"] = preview

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
