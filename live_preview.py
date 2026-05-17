import wx
import numpy as np
from PIL import Image


def clamp_preview_zoom(value):
    return max(0.25, min(3.0, round(value, 3)))


def next_preview_zoom(current_zoom, wheel_rotation):
    direction = 1 if wheel_rotation > 0 else -1
    factor = 1.15 if direction > 0 else (1 / 1.15)
    return clamp_preview_zoom(current_zoom * factor)


def compute_semantic_regions(
    *,
    image_height,
    render_success,
    latest_slice,
    matched_region,
    probe_region,
    static_top,
    static_bottom,
):
    latest_start, latest_end = latest_slice
    matched_start, matched_end = matched_region
    probe_start, probe_end = probe_region

    if matched_end > matched_start:
        footprint = (matched_start, min(image_height, matched_end))
    elif latest_end > latest_start:
        footprint = (latest_start, min(image_height, latest_end))
    else:
        footprint = (0, 0)

    if probe_end > probe_start:
        probe = (max(0, probe_start), min(image_height, probe_end))
    else:
        probe = (0, 0)

    static_top_band = (0, 0)
    static_bottom_band = (0, 0)
    if footprint[1] > footprint[0]:
        static_top_band = (
            footprint[0],
            min(footprint[1], footprint[0] + max(0, static_top)),
        )
        static_bottom_band = (
            max(footprint[0], footprint[1] - max(0, static_bottom)),
            footprint[1],
        )

    return {
        "footprint": footprint,
        "probe": probe,
        "show_latest_slice_color": render_success and footprint[1] > footprint[0],
        "show_previous_success_footprint": False,
        "static_top": static_top,
        "static_bottom": static_bottom,
        "static_top_band": static_top_band,
        "static_bottom_band": static_bottom_band,
    }


class LivePreviewFrame(wx.Frame):
    def __init__(self, screen_height, debug_mode=False, selection_region=None, manual_callback=None, undo_callback=None, cancel_callback=None):
        # Geometry defaults are refined after controls exist and displays are inspected.
        self.initial_width = 400 if not debug_mode else 500
        self.initial_height = 800  # Increased to fit new controls
        self.max_height = max(320, screen_height - 100)  # Recomputed from display bounds later
        self.preview_margin = 16
        self.min_preview_width_ratio = 0.40
        self.max_preview_width_ratio = 0.80
        self.min_preview_image_height = 160
        self.capture_source_height = max(1, screen_height)
        self.capture_source_width = selection_region['width'] if selection_region else self.initial_width
        self._render_image = None
        self._render_success = True
        self.preview_zoom = 1.0

        style = wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW | wx.CAPTION | wx.RESIZE_BORDER
        super().__init__(None, title="Live Stitcher", size=(self.initial_width, self.initial_height), style=style)

        self.debug_mode = debug_mode
        self.selection_region = selection_region
        self.manual_callback = manual_callback
        self.undo_callback = undo_callback
        self.cancel_callback = cancel_callback
        self._capture_hidden = False
        self._capture_restore_position = None

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.BLACK)

        # UI Elements
        self.sizer = wx.BoxSizer(wx.VERTICAL)


        # Control buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        if self.manual_callback:
            self.snap_button = wx.Button(self.panel, label="Take Screenshot")
            self.snap_button.Bind(wx.EVT_BUTTON, lambda evt: self.manual_callback())
            button_sizer.Add(self.snap_button, 1, wx.ALL | wx.EXPAND, 2)
        
        if self.undo_callback:
            self.undo_button = wx.Button(self.panel, label="Undo Last")
            self.undo_button.Bind(wx.EVT_BUTTON, lambda evt: self.undo_callback())
            button_sizer.Add(self.undo_button, 1, wx.ALL | wx.EXPAND, 2)

        if self.cancel_callback:
            self.cancel_button = wx.Button(self.panel, label="Cancel")
            self.cancel_button.Bind(wx.EVT_BUTTON, lambda evt: self.cancel_callback())
            button_sizer.Add(self.cancel_button, 1, wx.ALL | wx.EXPAND, 2)
        
        if button_sizer.GetChildren():
            self.sizer.Add(button_sizer, 0, wx.ALL | wx.EXPAND, 5)

        # Settings Panel (always visible)
        settings_panel = wx.Panel(self.panel)
        settings_panel.SetBackgroundColour(wx.Colour(50, 50, 50))
        settings_sizer = wx.BoxSizer(wx.VERTICAL)

        # Scroll Trigger Checkbox
        self.scroll_trigger_checkbox = wx.CheckBox(settings_panel, label="Auto-capture on scroll")
        self.scroll_trigger_checkbox.SetForegroundColour(wx.WHITE)
        self.scroll_trigger_checkbox.SetValue(True)  # Default enabled
        settings_sizer.Add(self.scroll_trigger_checkbox, 0, wx.ALL, 2)

        settings_panel.SetSizer(settings_sizer)
        self.sizer.Add(settings_panel, 0, wx.ALL | wx.EXPAND, 5)

        # Debug Info Panel (only if debug mode)
        if self.debug_mode:
            self.debug_panel = wx.Panel(self.panel)
            self.debug_panel.SetBackgroundColour(wx.Colour(40, 40, 40))
            debug_sizer = wx.BoxSizer(wx.VERTICAL)

            self.debug_title = wx.StaticText(self.debug_panel, label="Debug Statistics:")
            self.debug_title.SetForegroundColour(wx.YELLOW)
            debug_sizer.Add(self.debug_title, 0, wx.ALL, 2)

            self.debug_height = wx.StaticText(self.debug_panel, label="Total Height: 0px")
            self.debug_height.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(self.debug_height, 0, wx.ALL, 2)

            self.debug_added = wx.StaticText(self.debug_panel, label="Height Added: 0px")
            self.debug_added.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(self.debug_added, 0, wx.ALL, 2)

            self.debug_processing = wx.StaticText(self.debug_panel, label="Processing Time: 0.0s")
            self.debug_processing.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(self.debug_processing, 0, wx.ALL, 2)

            self.debug_debounce = wx.StaticText(self.debug_panel, label="Debounce Time: 0.0s")
            self.debug_debounce.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(self.debug_debounce, 0, wx.ALL, 2)

            self.debug_static_top = wx.StaticText(self.debug_panel, label="Static Top: 0px")
            self.debug_static_top.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(self.debug_static_top, 0, wx.ALL, 2)

            self.debug_static_bottom = wx.StaticText(self.debug_panel, label="Static Bottom: 0px")
            self.debug_static_bottom.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(self.debug_static_bottom, 0, wx.ALL, 2)

            # Tolerance Slider (debug only)
            tolerance_label = wx.StaticText(self.debug_panel, label="Tolerance:")
            tolerance_label.SetForegroundColour(wx.WHITE)
            debug_sizer.Add(tolerance_label, 0, wx.ALL, 2)
            
            self.tolerance_slider = wx.Slider(self.debug_panel, value=20, minValue=5, maxValue=50, 
                                            style=wx.SL_HORIZONTAL | wx.SL_LABELS)
            debug_sizer.Add(self.tolerance_slider, 0, wx.ALL | wx.EXPAND, 2)

            self.debug_panel.SetSizer(debug_sizer)
            self.sizer.Add(self.debug_panel, 0, wx.EXPAND | wx.ALL, 5)
        else:
            # Create tolerance slider for non-debug mode with default value
            self.tolerance_slider = wx.Slider(self.panel, value=20, minValue=5, maxValue=50)
            self.tolerance_slider.Hide()  # Hidden but accessible

        # Image Display Area
        self.image_ctrl = wx.StaticBitmap(self.panel)
        self.image_ctrl.Bind(wx.EVT_MOUSEWHEEL, self._on_preview_mousewheel)
        self.sizer.Add(self.image_ctrl, 1, wx.EXPAND | wx.ALL, 5)

        self.status_panel = wx.Panel(self.panel)
        self.status_panel.SetBackgroundColour(wx.Colour(28, 28, 28))
        status_sizer = wx.BoxSizer(wx.VERTICAL)
        self.status_text = wx.StaticText(self.status_panel, label="Capture: starting\nMerge: idle")
        self.status_text.SetForegroundColour(wx.WHITE)
        status_sizer.Add(self.status_text, 0, wx.ALL | wx.EXPAND, 6)
        self.status_panel.SetSizer(status_sizer)
        self.sizer.Add(self.status_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.panel.SetSizer(self.sizer)

        self.last_merged_image = None
        self.last_height_added = 0
        self.last_static_top = 0
        self.last_static_bottom = 0
        self.last_latest_slice_start = 0
        self.last_latest_slice_end = 0
        self.last_match_status = "none"
        self.last_matched_region_start = 0
        self.last_matched_region_end = 0
        self.last_overlap_visual_start = 0
        self.last_overlap_visual_end = 0
        self.Show()
        self.panel.Layout()
        self._apply_initial_geometry()

    def get_tolerance(self):
        """Get current tolerance value from slider"""
        return self.tolerance_slider.GetValue()

    def get_scroll_trigger_enabled(self):
        """Get current scroll trigger setting"""
        return self.scroll_trigger_checkbox.GetValue()

    def update_status(self, status_info):
        self.status_text.SetLabel(self._format_status(status_info))
        self.status_panel.Layout()
        self.panel.Layout()

    def _format_status(self, status_info):
        capture_state = status_info.get('capture_state', 'idle')
        merge_state = status_info.get('merge_state', 'idle')
        queue_size = status_info.get('queue_size', 0)
        queue_capacity = status_info.get('queue_capacity', 0)
        dims = status_info.get('total_dimensions', '0x0')
        merges = status_info.get('successful_merges', 0)
        next_capture = status_info.get('next_capture_in')
        merge_progress = status_info.get('merge_progress')
        merge_phase = status_info.get('merge_phase', '')
        last_result = status_info.get('last_result', '')
        recommendation = status_info.get('recommendation', '')

        next_capture_text = "next scroll"
        if next_capture is not None:
            next_capture_text = f"{max(0.0, next_capture):.2f}s"

        merge_progress_text = merge_state
        if merge_progress is not None:
            merge_progress_text = f"{merge_state} {int(max(0.0, min(1.0, merge_progress)) * 100)}%"
        if merge_phase:
            merge_progress_text = f"{merge_progress_text}, {merge_phase}"

        summary = recommendation or last_result or "Scroll until the color slice nears the bottom"

        return "\n".join([
            f"{capture_state} | {merge_progress_text}",
            f"{dims} stitched | queue {queue_size}/{queue_capacity} | next {next_capture_text}",
            summary,
        ])

    def _region_to_rect(self, region=None):
        region = region or self.selection_region
        if not region:
            return None
        return wx.Rect(region['left'], region['top'], region['width'], region['height'])

    def _rect_intersects(self, rect_a, rect_b):
        if rect_a is None or rect_b is None:
            return False
        return not (
            rect_a.x + rect_a.width <= rect_b.x or
            rect_b.x + rect_b.width <= rect_a.x or
            rect_a.y + rect_a.height <= rect_b.y or
            rect_b.y + rect_b.height <= rect_a.y
        )

    def _rect_within(self, inner_rect, outer_rect):
        return (
            inner_rect.x >= outer_rect.x and
            inner_rect.y >= outer_rect.y and
            inner_rect.x + inner_rect.width <= outer_rect.x + outer_rect.width and
            inner_rect.y + inner_rect.height <= outer_rect.y + outer_rect.height
        )

    def _build_rect(self, x, y, width, height):
        return wx.Rect(int(x), int(y), int(width), int(height))

    def _get_display_rects(self):
        rects = []
        for idx in range(wx.Display.GetCount()):
            rects.append(wx.Display(idx).GetClientArea())
        return rects

    def _get_display_bounds(self):
        rects = self._get_display_rects()
        if not rects:
            display_width, display_height = wx.DisplaySize()
            return wx.Rect(0, 0, display_width, display_height)

        min_x = min(rect.x for rect in rects)
        min_y = min(rect.y for rect in rects)
        max_right = max(rect.x + rect.width for rect in rects)
        max_bottom = max(rect.y + rect.height for rect in rects)
        return wx.Rect(min_x, min_y, max_right - min_x, max_bottom - min_y)

    def _get_frame_overhead(self):
        frame_size = self.GetSize()
        client_size = self.GetClientSize()
        frame_extra_w = max(0, frame_size.width - client_size.width)
        frame_extra_h = max(0, frame_size.height - client_size.height)

        controls_height = 0
        for child in self.panel.GetChildren():
            if child is self.image_ctrl:
                continue
            best = child.GetBestSize()
            controls_height += best.height
        controls_height += 30  # panel padding and inter-section spacing

        return frame_extra_w, frame_extra_h, controls_height

    def _get_non_image_client_height(self):
        controls_height = 0
        for child in self.panel.GetChildren():
            if child is self.image_ctrl:
                continue
            controls_height += child.GetSize().height
        return controls_height + 10

    def _fit_preview_image_size(self, available_width, available_height):
        selection_rect = self._region_to_rect()
        if selection_rect is None or selection_rect.width <= 0 or selection_rect.height <= 0:
            return None

        frame_extra_w, frame_extra_h, controls_height = self._get_frame_overhead()
        image_min_w = max(1, int(selection_rect.width * self.min_preview_width_ratio))
        image_max_w = min(
            int(selection_rect.width * self.max_preview_width_ratio),
            available_width - frame_extra_w - 12,
        )
        image_max_h = available_height - frame_extra_h - controls_height
        if image_max_w <= 0 or image_max_h <= 0:
            return None

        image_w = image_max_w
        if image_w < image_min_w and available_width - frame_extra_w - 12 > 0:
            image_w = available_width - frame_extra_w - 12
        if image_w <= 0:
            return None

        scale = image_w / selection_rect.width
        desired_source_visible_h = self.capture_source_height * 2
        image_h = int(desired_source_visible_h * scale)
        image_h = max(self.min_preview_image_height, min(image_h, image_max_h))

        window_w = image_w + frame_extra_w + 12
        window_h = image_h + frame_extra_h + controls_height
        return window_w, window_h, image_w * image_h

    def _find_safe_preview_rect(self):
        selection_rect = self._region_to_rect()
        if selection_rect is None:
            return None

        display_bounds = self._get_display_bounds()
        self.max_height = max(320, display_bounds.height - (self.preview_margin * 2))

        frame_extra_w, _, _ = self._get_frame_overhead()
        min_window_w = int(selection_rect.width * self.min_preview_width_ratio) + frame_extra_w + 12

        selection_top = selection_rect.y
        selection_right = selection_rect.x + selection_rect.width
        available_right = max(0, (display_bounds.x + display_bounds.width) - selection_right - (self.preview_margin * 2))
        available_left = max(0, selection_rect.x - display_bounds.x - (self.preview_margin * 2))

        if available_right >= min_window_w or available_left <= 0:
            side = "right"
        elif available_left >= min_window_w:
            side = "left"
        else:
            side = "right" if available_right >= available_left else "left"

        available_width = available_right if side == "right" else available_left
        available_height = display_bounds.height - (self.preview_margin * 2)
        fitted = self._fit_preview_image_size(available_width, available_height)
        if fitted is None:
            other_side = "left" if side == "right" else "right"
            other_width = available_left if side == "right" else available_right
            fitted = self._fit_preview_image_size(other_width, available_height)
            if fitted is None:
                return None
            side = other_side
            available_width = other_width

        window_w, window_h, _ = fitted
        if side == "right":
            pos_x = selection_right + self.preview_margin
        else:
            pos_x = selection_rect.x - window_w - self.preview_margin

        min_x = display_bounds.x + self.preview_margin
        max_x = (display_bounds.x + display_bounds.width) - self.preview_margin - window_w
        min_y = display_bounds.y + self.preview_margin
        max_y = (display_bounds.y + display_bounds.height) - self.preview_margin - window_h

        pos_x = max(min_x, min(pos_x, max_x))
        pos_y = max(min_y, min(selection_top, max_y))
        return wx.Rect(int(pos_x), int(pos_y), int(window_w), int(window_h))

    def ensure_safe_position(self):
        safe_rect = self._find_safe_preview_rect()
        if safe_rect is None:
            return False
        self.SetSize((safe_rect.width, safe_rect.height))
        self.SetPosition((safe_rect.x, safe_rect.y))
        return True

    def _apply_initial_geometry(self):
        if not self.ensure_safe_position():
            display_bounds = self._get_display_bounds()
            fallback_x = max(display_bounds.x + self.preview_margin, display_bounds.x + display_bounds.width - self.initial_width - self.preview_margin)
            fallback_y = display_bounds.y + self.preview_margin
            self.SetSize((self.initial_width, self.initial_height))
            self.SetPosition((fallback_x, fallback_y))

    def get_window_rect(self):
        return self.GetScreenRect()

    def overlaps_region(self, region=None):
        return self._rect_intersects(self.get_window_rect(), self._region_to_rect(region))

    def hide_for_capture(self):
        was_shown = self.IsShown()
        if was_shown:
            self._capture_restore_position = self.GetPosition()
            self.Hide()
            self._capture_hidden = True
        return was_shown

    def restore_after_capture(self, was_shown):
        if was_shown:
            if hasattr(self, "ShowWithoutActivating"):
                self.ShowWithoutActivating()
            else:
                self.Show()
            if self._capture_restore_position is not None:
                self.SetPosition(self._capture_restore_position)
            self._capture_hidden = False
            self._capture_restore_position = None

    def _on_preview_mousewheel(self, event):
        if not event.ControlDown():
            event.Skip()
            return

        self.preview_zoom = next_preview_zoom(self.preview_zoom, event.GetWheelRotation())
        self._draw_preview()

    def update_image(self, pil_image, status="Merged", success=True, debug_info=None):
        """
        Updates the preview with the BOTTOM part of the huge merged image.
        """
        self._render_image = pil_image.copy()
        self._render_success = success

        # Store overlay data
        if debug_info:
            if 'height_added' in debug_info:
                self.last_height_added = debug_info['height_added']
            if 'static_top' in debug_info:
                self.last_static_top = debug_info['static_top']
            if 'static_bottom' in debug_info:
                self.last_static_bottom = debug_info['static_bottom']
            if 'match_status' in debug_info:
                self.last_match_status = debug_info['match_status']
            if 'matched_region_start' in debug_info and debug_info['matched_region_start'] is not None:
                self.last_matched_region_start = debug_info['matched_region_start']
                self.last_matched_region_end = max(
                    self.last_matched_region_start,
                    debug_info.get('matched_region_end', self.last_matched_region_start),
                )
            if (
                'latest_slice_start' in debug_info and
                debug_info.get('latest_slice_end', 0) > debug_info['latest_slice_start']
            ):
                self.last_latest_slice_start = debug_info['latest_slice_start']
                self.last_latest_slice_end = debug_info['latest_slice_end']
            if (
                'overlap_visual_start' in debug_info and
                debug_info.get('overlap_visual_end', 0) > debug_info['overlap_visual_start']
            ):
                self.last_overlap_visual_start = debug_info['overlap_visual_start']
                self.last_overlap_visual_end = debug_info['overlap_visual_end']

        # Update debug info if provided
        if self.debug_mode and debug_info:
            self.debug_height.SetLabel(f"Total Height: {debug_info.get('total_height', pil_image.height)}px")
            self.debug_added.SetLabel(f"Height Added: {debug_info.get('height_added', self.last_height_added)}px")
            self.debug_processing.SetLabel(f"Processing Time: {debug_info.get('processing_time', 0.0):.3f}s")
            self.debug_debounce.SetLabel(f"Debounce Time: {debug_info.get('debounce_time', 0.0):.3f}s")
            self.debug_static_top.SetLabel(f"Static Top: {debug_info.get('static_top', self.last_static_top)}px")
            self.debug_static_bottom.SetLabel(f"Static Bottom: {debug_info.get('static_bottom', self.last_static_bottom)}px")

        self.panel.SetBackgroundColour(wx.BLACK)
        self._draw_preview()
        wx.CallLater(200, self.panel.Refresh)

    def _draw_preview(self):
        if self._render_image is None:
            return

        pil_image = self._render_image
        success = self._render_success
        w, h = pil_image.size
        available_image_w = max(1, self.GetClientSize().width - 10)
        client_height = max(1, self.GetClientSize().height - self._get_non_image_client_height())

        full_scale = min(self.max_preview_width_ratio, available_image_w / w, client_height / h)
        tail_scale_floor = self.min_preview_width_ratio

        if full_scale >= tail_scale_floor:
            scale = full_scale
            crop_top = 0
            crop = pil_image
            tail_mode = False
        else:
            scale = min(available_image_w / w, tail_scale_floor)
            if scale <= 0:
                scale = max(0.01, available_image_w / w)
            visible_source_h = max(1, int(client_height / scale))
            crop_h = min(h, visible_source_h)
            crop_top = h - crop_h
            crop = pil_image.crop((0, crop_top, w, crop_top + crop_h))
            tail_mode = True

        disp_w = max(1, int(crop.width * scale))
        disp_h = max(1, int(crop.height * scale))
        img_resized = crop.resize((disp_w, disp_h), Image.Resampling.BOX)

        regions = compute_semantic_regions(
            image_height=h,
            render_success=success,
            latest_slice=(self.last_latest_slice_start, self.last_latest_slice_end),
            matched_region=(self.last_matched_region_start, self.last_matched_region_end),
            probe_region=(self.last_overlap_visual_start, self.last_overlap_visual_end),
            static_top=self.last_static_top,
            static_bottom=self.last_static_bottom,
        )
        has_footprint = regions["footprint"][1] > regions["footprint"][0]
        has_probe = regions["probe"][1] > regions["probe"][0]
        if has_footprint or has_probe:
            img_with_overlay = img_resized.convert("L").convert("RGB")
        else:
            img_with_overlay = img_resized.copy()

        if has_footprint:
            img_with_overlay = self._color_region(
                img_with_overlay,
                img_resized,
                crop_top,
                scale,
                regions["footprint"],
            )
        if has_probe:
            img_with_overlay = self._add_match_overlay(
                img_with_overlay,
                crop_top,
                scale,
                regions["probe"],
                active=not success,
            )
            if self.last_static_top > 0 or self.last_static_bottom > 0:
                img_with_overlay = self._add_static_borders_overlay(
                    img_with_overlay,
                    crop_top,
                    scale,
                    regions["static_top_band"],
                    regions["static_bottom_band"],
                )
        if tail_mode:
            img_with_overlay = self._add_hidden_content_fade(img_with_overlay)

        img_with_overlay = self._apply_preview_zoom(img_with_overlay)

        wx_img = wx.Image(img_with_overlay.width, img_with_overlay.height)
        wx_img.SetData(img_with_overlay.convert("RGB").tobytes())
        bmp = wx_img.ConvertToBitmap()

        self.image_ctrl.SetBitmap(bmp)
        self.panel.Layout()
        self.panel.Refresh()

    def _add_hidden_content_fade(self, img_resized):
        from PIL import Image

        fade_height = min(36, img_resized.height)
        if fade_height <= 0:
            return img_resized

        overlay = Image.new("RGBA", img_resized.size, (0, 0, 0, 0))
        alpha_band = Image.new("L", (img_resized.width, fade_height))
        for y in range(fade_height):
            alpha = int(150 * (1 - (y / max(1, fade_height - 1))))
            for x in range(img_resized.width):
                alpha_band.putpixel((x, y), alpha)
        overlay.paste((0, 0, 0, 255), (0, 0, img_resized.width, fade_height), mask=alpha_band)
        return Image.alpha_composite(img_resized.convert("RGBA"), overlay).convert("RGB")

    def _apply_preview_zoom(self, img_resized):
        zoom = self.preview_zoom
        if abs(zoom - 1.0) < 0.001:
            return img_resized

        base_w, base_h = img_resized.size
        scaled_w = max(1, int(round(base_w * zoom)))
        scaled_h = max(1, int(round(base_h * zoom)))
        scaled = img_resized.resize((scaled_w, scaled_h), Image.Resampling.BOX)

        if zoom > 1.0:
            left = max(0, (scaled_w - base_w) // 2)
            top = max(0, scaled_h - base_h)
            return scaled.crop((left, top, left + base_w, top + base_h))

        canvas = Image.new("RGB", (base_w, base_h), (0, 0, 0))
        paste_x = max(0, (base_w - scaled_w) // 2)
        paste_y = max(0, base_h - scaled_h)
        canvas.paste(scaled, (paste_x, paste_y))
        return canvas

    def _color_region(self, base_img, color_source, crop_top, scale, region):
        region_start, region_end = region
        if region_end <= crop_top:
            return base_img

        visible_start = max(region_start, crop_top)
        visible_end = max(visible_start, min(region_end, crop_top + int(color_source.height / scale)))
        if visible_end <= visible_start:
            return base_img

        src_top = int((visible_start - crop_top) * scale)
        src_bottom = int((visible_end - crop_top) * scale)
        if src_bottom > src_top:
            color_band = color_source.crop((0, src_top, color_source.width, src_bottom))
            base_img.paste(color_band, (0, src_top))

        return base_img

    def _add_match_overlay(self, img_resized, crop_top, scale, region, active=False):
        from PIL import Image, ImageDraw

        match_start, match_end = region
        if match_end <= crop_top:
            return img_resized

        visible_start = max(match_start, crop_top)
        visible_end = max(visible_start, min(match_end, crop_top + int(img_resized.height / scale)))
        if visible_end <= visible_start:
            return img_resized

        band_top = int((visible_start - crop_top) * scale)
        band_bottom = int((visible_end - crop_top) * scale)
        overlay = Image.new("RGBA", img_resized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if active or self.last_match_status == "candidate":
            fill = (232, 178, 88, 26)
            edge = (244, 202, 132, 84)
        else:
            fill = (92, 164, 156, 18)
            edge = (132, 198, 190, 56)

        draw.rectangle([(0, band_top), (img_resized.width, band_bottom)], fill=fill)
        draw.line([(0, band_top), (img_resized.width, band_top)], fill=edge, width=1)
        draw.line([(0, max(band_top, band_bottom - 1)), (img_resized.width, max(band_top, band_bottom - 1))], fill=edge, width=1)
        return Image.alpha_composite(img_resized.convert("RGBA"), overlay).convert("RGB")

    def _add_static_borders_overlay(self, img_resized, crop_top, scale, static_top_band, static_bottom_band):
        from PIL import Image, ImageDraw

        overlay = Image.new('RGBA', img_resized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if static_top_band[1] <= static_top_band[0] and static_bottom_band[1] <= static_bottom_band[0]:
            return img_resized

        visible_limit = crop_top + int(img_resized.height / scale)

        def draw_band(start, end, fill, edge):
            visible_start = max(start, crop_top)
            visible_end = min(end, visible_limit)
            if visible_end <= visible_start:
                return

            band_top = int((visible_start - crop_top) * scale)
            band_bottom = int((visible_end - crop_top) * scale)
            draw.rectangle([(0, band_top), (img_resized.width, band_bottom)], fill=fill)
            draw.line([(0, band_top), (img_resized.width, band_top)], fill=edge, width=1)
            draw.line([(0, max(band_top, band_bottom - 1)), (img_resized.width, max(band_top, band_bottom - 1))], fill=edge, width=1)

        if static_top_band[1] > static_top_band[0]:
            draw_band(
                static_top_band[0],
                static_top_band[1],
                (0, 0, 0, 0),
                (214, 118, 94, 82),
            )

        if static_bottom_band[1] > static_bottom_band[0]:
            draw_band(
                static_bottom_band[0],
                static_bottom_band[1],
                (0, 0, 0, 0),
                (108, 146, 214, 82),
            )

        result = Image.alpha_composite(img_resized.convert('RGBA'), overlay)
        return result.convert('RGB')
