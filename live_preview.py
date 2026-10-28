import itertools
import math

import wx
import wx.lib.buttons
import numpy as np
from PIL import Image, ImageDraw


def clamp_preview_zoom(value):
    return max(0.25, min(3.0, round(value, 3)))


def next_preview_zoom(current_zoom, wheel_rotation):
    direction = 1 if wheel_rotation > 0 else -1
    factor = 1.15 if direction > 0 else (1 / 1.15)
    return clamp_preview_zoom(current_zoom * factor)


def describe_preview_region(pointer_y, footprint, probe, static_top_band, static_bottom_band):
    if probe[0] <= pointer_y < probe[1]:
        return "matched probe used for alignment"
    if static_top_band[0] <= pointer_y < static_top_band[1]:
        return "ignored static top border"
    if static_bottom_band[0] <= pointer_y < static_bottom_band[1]:
        return "ignored static bottom border"
    if footprint[0] <= pointer_y < footprint[1]:
        return "relevant screenshot footprint"
    return "stitched history"


def compute_semantic_regions(
    *,
    image_height,
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
        "static_top": static_top,
        "static_bottom": static_bottom,
        "static_top_band": static_top_band,
        "static_bottom_band": static_bottom_band,
    }


# The window is a strip-chart recorder: the stitch prints on chart paper that feeds up past the pen carriage at
# the foot, the left margin carries the px scale and the match-score pen trace, the housing holds the keys
PAPER = (223, 232, 214)
GRID_MAJOR = (159, 181, 154)
GRID_MINOR = (200, 214, 192)
SCALE_INK = wx.Colour(43, 58, 57)
HOUSING = wx.Colour(43, 52, 57)
KEY_FACE = wx.Colour(47, 55, 62)
KEY_EDGE = wx.Colour(86, 96, 107)
KEY_GO = wx.Colour(25, 130, 72)
HOUSING_TEXT = wx.Colour(236, 239, 234)
HOUSING_MUTED = wx.Colour(150, 156, 150)
PEN = (75, 45, 127)
EVENT = (192, 57, 43)
# Outline colour of the region selector and capture veil, which keep their own look
ACCENT = (92, 164, 156)
PAD = 8
GAP = 4
GRID_COLUMN = 10  # Vertical paper rules, in window px
TRACE_W = 36
MARGIN_GAP = 20
MARGIN_RIGHT = 36
# The trace folds away before the stitch drops under this width
MIN_STITCH_W_WITH_TRACE = 220
LABEL_SPACING = 48  # Minimum window px between scale numerals
SCALE_STEPS = (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000)
TRACE_FLOOR = 0.4  # Scores below this never merge; the trace's left edge
ADVANCE_MS = 220
GRIP = 14  # Resize grip at the carriage's right end
MIN_WINDOW = (300, 360)
AUTO_SCROLL_LABELS = {False: "Auto Scroll", True: "Stop Auto Scroll"}
ZOOM_TIP = "Ctrl+scroll to zoom, double-click to reset"


def _style_key(button, face=KEY_FACE, widest_label=None):
    # Sized to the label rather than the platform's default button width, so the strip holds one row
    text_w, text_h = button.GetTextExtent(widest_label or button.GetLabel())
    button.SetMinSize((text_w + PAD * 3, text_h + PAD * 2))
    button.SetBezelWidth(1)
    button.SetBackgroundColour(face)
    button.SetForegroundColour(HOUSING_TEXT)
    # Keys are flat engraved plates: one even edge, which turns to pen ink while pressed or latched
    def draw_edge(dc, x1, y1, x2, y2):
        dc.SetPen(wx.Pen(KEY_EDGE if button.up else wx.Colour(*PEN).ChangeLightness(160), 1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
    button.DrawBezel = draw_edge


def _scale_step(scale):
    return next((step for step in SCALE_STEPS if step * scale >= LABEL_SPACING), SCALE_STEPS[-1])


def _animations_enabled():
    if wx.Platform == "__WXMSW__":
        import ctypes
        enabled = ctypes.c_bool()
        # SPI_GETCLIENTAREAANIMATION, the "Show animations in Windows" setting
        if not ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
            raise ctypes.WinError()
        return enabled.value
    # GTK exposes no reduced-motion setting to wx; the advance is short and only shows where content went
    return True


class LivePreviewFrame(wx.Frame):
    def __init__(self, screen_height, *, debug_mode, selection_region, manual_callback, undo_callback, cancel_callback, done_callback, veil_callback, auto_scroll_callback):
        # Geometry defaults are refined after controls exist and displays are inspected.
        self.initial_width = 400 if not debug_mode else 500
        self.initial_height = 800
        self.max_height = max(320, screen_height - 100)  # Recomputed from display bounds later
        self.preview_margin = 16
        self.min_preview_width_ratio = 0.40
        self.max_preview_width_ratio = 0.80
        self.min_preview_image_height = 160
        self.capture_source_height = max(1, screen_height)
        self.capture_source_width = selection_region['width'] if selection_region else self.initial_width
        self._render_image = None
        self._render_success = True
        # Region metadata of the shown merge; replaced wholesale so undo and failures never mix two merges
        self._overlay = {}
        # Pen trace: one entry per merge attempt, in stitch coordinates; undone merges stay, struck through
        self._trace = []
        self.preview_zoom = 1.0
        # Source-pixel origin of a zoomed view; None follows the newest content at the bottom
        self._zoom_origin = None
        self._view = None
        self._paper_bitmap = None
        self._advance = None
        self._animate = _animations_enabled()
        self._hover_label = None
        self._carriage_text = ("", "")
        self._carriage_failed = False
        self._last_status_info = None
        self._shown_status = None
        self._scroll_trigger_enabled = True

        # Frameless: the housing is the window chrome, dragged by its key strip and resized from the carriage grip
        style = wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW | wx.BORDER_NONE
        super().__init__(None, title="Emingle", size=(self.initial_width, self.initial_height), style=style)

        self.debug_mode = debug_mode
        self.selection_region = selection_region
        self.manual_callback = manual_callback
        self.undo_callback = undo_callback
        self.cancel_callback = cancel_callback
        self.done_callback = done_callback
        self.veil_callback = veil_callback
        self.auto_scroll_callback = auto_scroll_callback
        self._capture_hidden = False
        self._capture_restore_position = None
        self._last_hover_regions = None

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(HOUSING)
        self._drag = None

        self.title_label = wx.StaticText(self.panel, label="Emingle")
        self.title_label.SetForegroundColour(HOUSING_TEXT)
        self.title_label.SetFont(self.title_label.GetFont().Larger().Bold())
        for widget in (self.panel, self.title_label):
            widget.Bind(wx.EVT_LEFT_DOWN, lambda evt, widget=widget: self._start_drag(widget, "move"))
            widget.Bind(wx.EVT_MOTION, self._on_drag_motion)
            widget.Bind(wx.EVT_LEFT_UP, self._end_drag)
            widget.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_drag_lost)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Housing front panel: the five keys spanning one row, then the options menu key

        self.snap_button = wx.lib.buttons.GenButton(self.panel, label="Take Screenshot")
        _style_key(self.snap_button)
        self.snap_button.Bind(wx.EVT_BUTTON, lambda evt: self.manual_callback())

        self.auto_scroll_button = wx.lib.buttons.GenToggleButton(self.panel, label=AUTO_SCROLL_LABELS[False])
        _style_key(self.auto_scroll_button, widest_label=AUTO_SCROLL_LABELS[True])
        self.auto_scroll_button.Bind(wx.EVT_BUTTON, self._on_auto_scroll_toggled)

        self.undo_button = wx.lib.buttons.GenButton(self.panel, label="Undo")
        _style_key(self.undo_button)
        self.undo_button.Bind(wx.EVT_BUTTON, lambda evt: self.undo_callback())

        self.cancel_button = wx.lib.buttons.GenButton(self.panel, label="Cancel")
        _style_key(self.cancel_button)
        self.cancel_button.Bind(wx.EVT_BUTTON, lambda evt: self.cancel_callback())

        self.done_button = wx.lib.buttons.GenButton(self.panel, label="Done")
        _style_key(self.done_button, KEY_GO)
        self.done_button.SetToolTip("Copy the stitched image and close (Esc)")
        self.done_button.Bind(wx.EVT_BUTTON, lambda evt: self.done_callback())

        self.menu_button = wx.lib.buttons.GenBitmapButton(self.panel, bitmap=self._menu_glyph())
        _style_key(self.menu_button)
        self.menu_button.SetToolTip("Options")
        self.menu_button.Bind(wx.EVT_BUTTON, self._on_menu)

        self._keys = [self.snap_button, self.auto_scroll_button, self.undo_button, self.cancel_button, self.done_button, self.menu_button]
        self._key_rows = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self._key_rows, 0, wx.EXPAND | wx.ALL, PAD)
        self._key_layout = None
        self._arrange_keys(self._key_rows_for(self.initial_width))

        self.options_menu = wx.Menu()
        self.scroll_trigger_item = self.options_menu.AppendCheckItem(wx.ID_ANY, "Auto-capture on scroll")
        self.scroll_trigger_item.Check(True)
        self.veil_item = self.options_menu.AppendCheckItem(wx.ID_ANY, "Dim outside the capture area")
        self.veil_item.Check(True)
        self.Bind(wx.EVT_MENU, self._on_scroll_trigger_toggled, self.scroll_trigger_item)
        self.Bind(wx.EVT_MENU, self._on_veil_toggled, self.veil_item)

        # Debug Info Panel (only if debug mode)
        if self.debug_mode:
            self.debug_panel = wx.Panel(self.panel)
            self.debug_panel.SetBackgroundColour(KEY_FACE)
            debug_sizer = wx.BoxSizer(wx.VERTICAL)

            self.debug_status = wx.StaticText(self.debug_panel, label="", style=wx.ST_ELLIPSIZE_END | wx.ST_NO_AUTORESIZE)
            self.debug_status.SetForegroundColour(HOUSING_TEXT)
            self.debug_status.SetFont(self.debug_status.GetFont().Bold())
            debug_sizer.Add(self.debug_status, 0, wx.ALL | wx.EXPAND, 2)

            self.debug_height = wx.StaticText(self.debug_panel, label="Total Height: 0px")
            self.debug_height.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(self.debug_height, 0, wx.ALL, 2)

            self.debug_added = wx.StaticText(self.debug_panel, label="Height Added: 0px")
            self.debug_added.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(self.debug_added, 0, wx.ALL, 2)

            self.debug_processing = wx.StaticText(self.debug_panel, label="Processing Time: 0.0s")
            self.debug_processing.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(self.debug_processing, 0, wx.ALL, 2)

            self.debug_debounce = wx.StaticText(self.debug_panel, label="Debounce Time: 0.0s")
            self.debug_debounce.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(self.debug_debounce, 0, wx.ALL, 2)

            self.debug_static_top = wx.StaticText(self.debug_panel, label="Static Top: 0px")
            self.debug_static_top.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(self.debug_static_top, 0, wx.ALL, 2)

            self.debug_static_bottom = wx.StaticText(self.debug_panel, label="Static Bottom: 0px")
            self.debug_static_bottom.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(self.debug_static_bottom, 0, wx.ALL, 2)

            # Tolerance Slider (debug only)
            tolerance_label = wx.StaticText(self.debug_panel, label="Tolerance:")
            tolerance_label.SetForegroundColour(HOUSING_MUTED)
            debug_sizer.Add(tolerance_label, 0, wx.ALL, 2)

            self.tolerance_slider = wx.Slider(self.debug_panel, value=20, minValue=5, maxValue=50,
                                            style=wx.SL_HORIZONTAL | wx.SL_LABELS)
            debug_sizer.Add(self.tolerance_slider, 0, wx.ALL | wx.EXPAND, 2)

            debug_frame = wx.BoxSizer(wx.VERTICAL)
            debug_frame.Add(debug_sizer, 1, wx.ALL | wx.EXPAND, PAD - 2)
            self.debug_panel.SetSizer(debug_frame)
            self.sizer.Add(self.debug_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, PAD)
        else:
            # Create tolerance slider for non-debug mode with default value
            self.tolerance_slider = wx.Slider(self.panel, value=20, minValue=5, maxValue=50)
            self.tolerance_slider.Hide()  # Hidden but accessible

        # Chart paper, edge to edge: painted from one bitmap the size of the panel, so pointer coordinates map
        # straight onto the rendering and the bitmap never feeds its own size back into the sizer
        self.paper = wx.Panel(self.panel)
        self.paper.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.paper.SetMinSize((1, 1))
        self.paper.SetToolTip(ZOOM_TIP)
        self.paper.Bind(wx.EVT_PAINT, self._on_paper_paint)
        self.paper.Bind(wx.EVT_SIZE, self._on_paper_resize)
        self.paper.Bind(wx.EVT_MOUSEWHEEL, self._on_paper_mousewheel)
        self.paper.Bind(wx.EVT_LEFT_DCLICK, self._on_paper_reset_zoom)
        self.paper.Bind(wx.EVT_MOTION, self._on_paper_hover)
        self.sizer.Add(self.paper, 1, wx.EXPAND)
        # Tabular numerals a step below the body size, so columns of figures line up and hold a narrow margin
        self._scale_font = wx.Font(wx.FontInfo(self.paper.GetFont().GetPointSize() - 1).Family(wx.FONTFAMILY_TELETYPE))
        # Wide enough for a five-digit numeral, its gap and the major tick
        probe_dc = wx.ClientDC(self.paper)
        probe_dc.SetFont(self._scale_font)
        self._scale_w = probe_dc.GetTextExtent("88,888").width + GAP * 3 + 7
        self._redraw_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._draw_preview(), self._redraw_timer)
        self._advance_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda evt: self._draw_preview(), self._advance_timer)

        # Pen carriage across the foot: the pen marker and one line of status printed on it
        self.carriage = wx.Panel(self.panel)
        self.carriage.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.carriage.SetMinSize((-1, self.carriage.GetCharHeight() + PAD * 2 + GAP))
        self.carriage.Bind(wx.EVT_PAINT, self._on_carriage_paint)
        self.carriage.Bind(wx.EVT_SIZE, lambda evt: (self.carriage.Refresh(), evt.Skip()))
        self.carriage.Bind(wx.EVT_LEFT_DOWN, self._on_carriage_down)
        self.carriage.Bind(wx.EVT_MOTION, self._on_carriage_motion)
        self.carriage.Bind(wx.EVT_LEFT_UP, self._end_drag)
        self.carriage.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_drag_lost)
        self.sizer.Add(self.carriage, 0, wx.EXPAND)

        self.panel.Bind(wx.EVT_SIZE, self._on_panel_resize)
        self.panel.SetSizer(self.sizer)

        self.Show()
        self.panel.Layout()
        self._apply_initial_geometry()

    def _key_rows_for(self, width):
        # Fewest rows that fit, split so the rows carry even widths: the name leads the first, the options key
        # closes the last, and every key keeps a key's width instead of one stretching across a row of its own
        keys = self._keys[:-1]
        widths = [key.GetMinSize().width + GAP for key in keys]
        lead = self.title_label.GetBestSize().width + PAD * 3
        tail = self.menu_button.GetMinSize().width
        available = width - PAD * 2
        for count in range(1, len(keys) + 1):
            best = None
            for cuts in itertools.combinations(range(1, len(keys)), count - 1):
                bounds = (0, *cuts, len(keys))
                row_widths = [sum(widths[a:b]) for a, b in zip(bounds, bounds[1:])]
                row_widths[0] += lead
                row_widths[-1] += tail
                if max(row_widths) <= available and (best is None or max(row_widths) < best[0]):
                    best = (max(row_widths), bounds)
            if best is not None:
                bounds = best[1]
                break
        else:
            bounds = tuple(range(len(keys) + 1))
        rows = [list(keys[a:b]) for a, b in zip(bounds, bounds[1:])]
        rows[0].insert(0, self.title_label)
        rows[-1].append(self.menu_button)
        return rows

    def _arrange_keys(self, rows):
        self._key_layout = [len(row) for row in rows]
        self._key_rows.Clear(delete_windows=False)
        for index, items in enumerate(rows):
            row = wx.BoxSizer(wx.HORIZONTAL)
            for item in items:
                if item is self.title_label:
                    row.Add(item, 0, wx.ALIGN_CENTER_VERTICAL | (wx.RIGHT if item is not items[-1] else 0), PAD * 3)
                else:
                    row.Add(item, 0 if item is self.menu_button else 1, wx.EXPAND | (wx.RIGHT if item is not items[-1] else 0), GAP)
            self._key_rows.Add(row, 0, wx.EXPAND | (wx.BOTTOM if index < len(rows) - 1 else 0), GAP)

    def _on_panel_resize(self, event):
        event.Skip()
        rows = self._key_rows_for(event.GetSize().width)
        if [len(row) for row in rows] != self._key_layout:
            self._arrange_keys(rows)
        self._fit_carriage(event.GetSize().width)

    def _start_drag(self, widget, mode):
        self._drag = {"mode": mode, "widget": widget, "origin": wx.GetMousePosition(), "rect": self.GetRect()}
        widget.CaptureMouse()

    def _on_drag_motion(self, event):
        event.Skip()
        if self._drag is None:
            return
        delta = wx.GetMousePosition() - self._drag["origin"]
        rect = self._drag["rect"]
        if self._drag["mode"] == "move":
            self.SetPosition(rect.GetTopLeft() + delta)
        else:
            self.SetSize(max(MIN_WINDOW[0], rect.width + delta.x), max(MIN_WINDOW[1], rect.height + delta.y))

    def _end_drag(self, event):
        if self._drag is not None and self._drag["widget"].HasCapture():
            self._drag["widget"].ReleaseMouse()
        self._drag = None

    def _on_drag_lost(self, event):
        self._drag = None

    def _in_grip(self, position):
        width, height = self.carriage.GetClientSize()
        return position.x >= width - GRIP - GAP and position.y >= height - GRIP - GAP

    def _on_carriage_down(self, event):
        if self._in_grip(event.GetPosition()):
            self._start_drag(self.carriage, "resize")
        else:
            self._start_drag(self.carriage, "move")

    def _on_carriage_motion(self, event):
        self.carriage.SetCursor(wx.Cursor(wx.CURSOR_SIZENWSE if self._in_grip(event.GetPosition()) or (self._drag and self._drag["mode"] == "resize") else wx.CURSOR_ARROW))
        self._on_drag_motion(event)

    def _menu_glyph(self):
        # Three dots drawn on the key face, sized to the key's text height
        size = self.GetCharHeight()
        bitmap = wx.Bitmap(size, size)
        dc = wx.MemoryDC(bitmap)
        dc.SetBackground(wx.Brush(KEY_FACE))
        dc.Clear()
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush(HOUSING_TEXT))
        radius = max(1, size // 10)
        for i in (-1, 0, 1):
            dc.DrawCircle(size // 2 + i * radius * 3, size // 2, radius)
        dc.SelectObject(wx.NullBitmap)
        return bitmap

    def get_tolerance(self):
        """Get current tolerance value from slider"""
        return self.tolerance_slider.GetValue()

    def get_scroll_trigger_enabled(self):
        # Read from the capture thread, so it returns a plain bool rather than querying a widget
        return self._scroll_trigger_enabled

    def set_auto_scroll(self, enabled):
        self.auto_scroll_button.SetToggle(enabled)
        self.auto_scroll_button.SetLabel(AUTO_SCROLL_LABELS[enabled])

    def _on_auto_scroll_toggled(self, event):
        # The label names what pressing it does next, so a running auto scroll reads as stoppable
        self.auto_scroll_button.SetLabel(AUTO_SCROLL_LABELS[event.GetIsDown()])
        self.auto_scroll_callback(event.GetIsDown())

    def _on_menu(self, event):
        button = self.menu_button
        self.PopupMenu(self.options_menu, button.GetPosition() + wx.Point(0, button.GetSize().height))

    def _on_scroll_trigger_toggled(self, event):
        self._scroll_trigger_enabled = event.IsChecked()

    def _on_veil_toggled(self, event):
        self.veil_callback(event.IsChecked())
        # The veil is always on top too; showing it again would bury this window
        self.Raise()

    def trace_merge(self, y, score):
        self._trace.append({"y": y, "score": score, "state": "merged"})
        self.paper.Refresh()

    def trace_failure(self, y, score):
        self._trace.append({"y": y, "score": score, "state": "failed"})
        self.paper.Refresh()

    def trace_undo(self):
        merged = [entry for entry in self._trace if entry["state"] == "merged"]
        merged[-1]["state"] = "undone"
        self.paper.Refresh()

    def update_status(self, status_info):
        self._last_status_info = status_info
        title, metrics, result, hint = self._format_status(status_info)
        failed = status_info["last_result_failed"]
        accepting = status_info["accepting_input"]
        controls = (accepting, status_info["queue_size"] < status_info["queue_capacity"], status_info["can_undo"])
        # Status is published several times a second; only repaint when something visible changed
        shown = (title, metrics, result, failed, hint, controls)
        if shown == self._shown_status:
            return
        self._shown_status = shown

        if status_info["merge_state"] == "merging":
            result = f"Merging {int(max(0.0, min(1.0, status_info['merge_progress'])) * 100)}%"
        self._carriage_text = (result, hint)
        self._carriage_failed = failed
        self.carriage.SetToolTip(f"{result}\n{hint}" if result else hint)
        self._fit_carriage(self.panel.GetClientSize().width)
        self.carriage.Refresh()

        if self.debug_mode:
            self.debug_status.SetLabel(f"{title}  ·  {metrics}")

        self.snap_button.Enable(accepting and controls[1])
        self.auto_scroll_button.Enable(accepting)
        self.undo_button.Enable(accepting and controls[2])
        self.cancel_button.Enable(accepting)
        self.done_button.Enable(accepting)

    def _format_status(self, status_info):
        capture_state = status_info["capture_state"]
        title = f"{capture_state[:1].upper()}{capture_state[1:]}"
        if status_info["merge_state"] == "merging":
            title = f"{title}  ·  merging {int(max(0.0, min(1.0, status_info['merge_progress'])) * 100)}%"

        metrics = [
            f"{status_info['total_dimensions'].replace('x', ' × ')} px",
            f"{status_info['successful_merges']} merged",
            f"queue {status_info['queue_size']}/{status_info['queue_capacity']}",
        ]
        if status_info["next_capture_in"] is not None:
            metrics.append(f"next capture in {max(0.0, status_info['next_capture_in']):.1f}s")
        if status_info["merge_phase"]:
            metrics.append(status_info["merge_phase"])

        # A failed result keeps its own advice until the next merge, unless the capture loop has something more urgent
        if status_info["last_result_failed"] and capture_state == "waiting for scroll":
            hint = status_info["failure_hint"]
        else:
            hint = status_info["recommendation"]

        return title, "  ·  ".join(metrics), status_info["last_result"], hint

    def _carriage_layout(self, dc, available):
        # Result and advice share one line when they fit; otherwise the advice takes its own lines, wrapped whole
        result, hint = self._carriage_text
        regular, bold = self.carriage.GetFont(), self.carriage.GetFont().Bold()
        dc.SetFont(bold)
        result_w = dc.GetTextExtent(result).width
        dc.SetFont(regular)
        if result and result_w + PAD + dc.GetTextExtent(hint).width <= available:
            return [[(result, bold), (hint, regular)]]
        lines = [[(result, bold)]] if result else []
        current = ""
        for word in hint.split():
            candidate = f"{current} {word}".strip()
            if current and dc.GetTextExtent(candidate).width > available:
                lines.append([(current, regular)])
                current = word
            else:
                current = candidate
        lines.append([(current, regular)])
        return lines

    def _carriage_text_area(self, width):
        pen_x = self._paper_margin_left() + MARGIN_GAP
        text_x = pen_x + self.carriage.GetCharHeight() // 2 + PAD
        return pen_x, text_x, max(1, width - GRIP - GAP * 2 - text_x)

    def _fit_carriage(self, width):
        dc = wx.ClientDC(self.carriage)
        lines = len(self._carriage_layout(dc, self._carriage_text_area(width)[2]))
        height = self.carriage.GetCharHeight() * lines + GAP * (lines - 1) + PAD * 2 + GAP
        if self.carriage.GetMinSize().height != height:
            self.carriage.SetMinSize((-1, height))
            self.panel.Layout()

    def _on_carriage_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self.carriage)
        dc.SetBackground(wx.Brush(HOUSING))
        dc.Clear()
        width, height = self.carriage.GetClientSize()
        char_h = self.carriage.GetCharHeight()
        ink = wx.Colour(*(EVENT if self._carriage_failed else PEN))

        # The pen sits where the stitch column starts, nib up toward the paper, like the recorder's pen
        pen_x, text_x, available = self._carriage_text_area(width)
        top = PAD + GAP // 2
        half = max(3, char_h // 4)
        gc = wx.GraphicsContext.Create(dc)
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.SetBrush(wx.Brush(ink.ChangeLightness(140)))
        nib = gc.CreatePath()
        nib.MoveToPoint(pen_x, top)
        nib.AddLineToPoint(pen_x + half, top + char_h * 0.45)
        nib.AddLineToPoint(pen_x + half, top + char_h)
        nib.AddLineToPoint(pen_x - half, top + char_h)
        nib.AddLineToPoint(pen_x - half, top + char_h * 0.45)
        nib.CloseSubpath()
        gc.FillPath(nib)
        gc.SetPen(wx.Pen(HOUSING, 1))
        gc.StrokeLine(pen_x, top + 2, pen_x, top + char_h * 0.55)

        # Resize grip: three short diagonal rules in the corner
        dc.SetPen(wx.Pen(KEY_EDGE, 1))
        for offset in (4, 8, 12):
            dc.DrawLine(width - GAP - offset, height - GAP, width - GAP, height - GAP - offset)

        for index, line in enumerate(self._carriage_layout(dc, available)):
            x = text_x
            y = top + index * (char_h + GAP)
            for text, font in line:
                dc.SetFont(font)
                bold = font.GetWeight() == wx.FONTWEIGHT_BOLD
                dc.SetTextForeground(ink.ChangeLightness(170) if bold and self._carriage_failed else HOUSING_TEXT)
                text = wx.Control.Ellipsize(text, dc, wx.ELLIPSIZE_END, max(1, text_x + available - x))
                dc.DrawText(text, x, y)
                x += dc.GetTextExtent(text).width + PAD

    def _paper_margin_left(self):
        width = self.paper.GetClientSize().width
        with_trace = self._scale_w + TRACE_W + MARGIN_GAP
        if width - with_trace - MARGIN_RIGHT >= MIN_STITCH_W_WITH_TRACE:
            return with_trace
        return self._scale_w

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
        # Window decorations plus the paper margins around the stitch column
        frame_extra_w = max(0, frame_size.width - client_size.width) + self._scale_w + TRACE_W + MARGIN_GAP + MARGIN_RIGHT
        frame_extra_h = max(0, frame_size.height - client_size.height)

        # The sizer already knows the housing rows; the paper contributes its 1px minimum
        controls_height = self.sizer.CalcMin().height - 1

        return frame_extra_w, frame_extra_h, controls_height

    def _fit_preview_image_size(self, available_width, available_height):
        selection_rect = self._region_to_rect()
        if selection_rect is None or selection_rect.width <= 0 or selection_rect.height <= 0:
            return None

        frame_extra_w, frame_extra_h, controls_height = self._get_frame_overhead()
        image_min_w = max(1, int(selection_rect.width * self.min_preview_width_ratio))
        image_max_w = min(
            int(selection_rect.width * self.max_preview_width_ratio),
            available_width - frame_extra_w,
        )
        image_max_h = available_height - frame_extra_h - controls_height
        if image_max_w <= 0 or image_max_h <= 0:
            return None

        image_w = image_max_w
        if image_w < image_min_w and available_width - frame_extra_w > 0:
            image_w = available_width - frame_extra_w
        if image_w <= 0:
            return None

        scale = image_w / selection_rect.width
        desired_source_visible_h = self.capture_source_height * 2
        image_h = int(desired_source_visible_h * scale)
        image_h = max(self.min_preview_image_height, min(image_h, image_max_h))

        window_w = image_w + frame_extra_w
        window_h = image_h + frame_extra_h + controls_height
        return window_w, window_h, image_w * image_h

    def _find_safe_preview_rect(self):
        selection_rect = self._region_to_rect()
        if selection_rect is None:
            return None

        display_bounds = self._get_display_bounds()
        self.max_height = max(320, display_bounds.height - (self.preview_margin * 2))

        frame_extra_w, _, _ = self._get_frame_overhead()
        min_window_w = int(selection_rect.width * self.min_preview_width_ratio) + frame_extra_w

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

    def _on_paper_resize(self, event):
        event.Skip()
        # Dragging the window border fires a burst of size events; render once it settles
        self._redraw_timer.StartOnce(60)
        self.carriage.Refresh()

    def _on_paper_mousewheel(self, event):
        if not event.ControlDown():
            event.Skip()
            return
        if self._view is None:
            return  # Nothing rendered yet

        base_scale, scale, origin_x, origin_y = self._view
        stitch_left = self._paper_margin_left() + MARGIN_GAP
        pointer = event.GetPosition()
        # Keep the source pixel under the pointer in place while the scale changes
        pointer_x = pointer.x - stitch_left
        anchor_x = origin_x + pointer_x / scale
        anchor_y = origin_y + pointer.y / scale
        self.preview_zoom = next_preview_zoom(self.preview_zoom, event.GetWheelRotation())
        self._advance = None
        if abs(self.preview_zoom - 1.0) < 0.001:
            self._zoom_origin = None
        else:
            new_scale = base_scale * self.preview_zoom
            self._zoom_origin = (anchor_x - pointer_x / new_scale, anchor_y - pointer.y / new_scale)
        self._draw_preview()

    def _on_paper_reset_zoom(self, event):
        self.preview_zoom = 1.0
        self._zoom_origin = None
        self._draw_preview()

    def _on_paper_hover(self, event):
        event.Skip()
        if self._last_hover_regions is None:
            return

        label = describe_preview_region(
            pointer_y=event.GetPosition().y,
            footprint=self._last_hover_regions["footprint"],
            probe=self._last_hover_regions["probe"],
            static_top_band=self._last_hover_regions["static_top_band"],
            static_bottom_band=self._last_hover_regions["static_bottom_band"],
        )
        if label != self._hover_label:
            self._hover_label = label
            self.paper.SetToolTip(f"{label[:1].upper()}{label[1:]}\n{ZOOM_TIP}")

    def update_image(self, pil_image, success=True, overlay=None):
        """
        Shows the merged image. `overlay` replaces the region metadata; None keeps the current one, which is how a
        capture without any match leaves the last meaningful context on screen.
        """
        previous = self._render_image
        self._render_image = pil_image
        self._render_success = success
        if overlay is not None:
            self._overlay = overlay

        if self.debug_mode and overlay:
            self.debug_height.SetLabel(f"Total Height: {overlay.get('total_height', pil_image.height)}px")
            self.debug_added.SetLabel(f"Height Added: {overlay.get('height_added', 0)}px")
            self.debug_processing.SetLabel(f"Processing Time: {overlay.get('processing_time', 0.0):.3f}s")
            self.debug_debounce.SetLabel(f"Debounce Time: {overlay.get('debounce_time', 0.0):.3f}s")
            self.debug_static_top.SetLabel(f"Static Top: {overlay.get('static_top', 0)}px")
            self.debug_static_bottom.SetLabel(f"Static Bottom: {overlay.get('static_bottom', 0)}px")

        # Signature motion: when the stitch grows while following the tail, the paper feeds up past the carriage
        # from where it was, so the eye sees where the new panel came in
        grew = previous is not None and pil_image.height > previous.height
        if self._animate and grew and success and self._zoom_origin is None and self._view is not None:
            base_scale, _, _, origin_y = self._view
            self._advance = {"from": origin_y, "base_scale": base_scale, "started": wx.GetUTCTimeMillis()}
            self._advance_timer.Start(16)

        self._draw_preview()

    def _draw_preview(self):
        if self._render_image is None:
            return

        image = self._render_image
        w, h = image.size
        paper_w, paper_h = self.paper.GetClientSize()
        paper_w, paper_h = max(1, paper_w), max(1, paper_h)
        stitch_left = self._paper_margin_left() + MARGIN_GAP
        slot_w = max(1, paper_w - stitch_left - MARGIN_RIGHT)
        slot_h = paper_h

        # At 1x the whole stitch shows when it fits at a readable scale, otherwise its newest tail does
        full_scale = min(self.max_preview_width_ratio, slot_w / w, slot_h / h)
        if full_scale >= self.min_preview_width_ratio:
            base_scale = full_scale
        else:
            # Too tall to fit: the tail fills the stitch column so the newest panel reads at full width
            base_scale = min(slot_w / w, self.max_preview_width_ratio)
        scale = base_scale * self.preview_zoom
        view_w, view_h = slot_w / scale, slot_h / scale

        # The view origin is in source pixels, so zooming resamples the stitch itself instead of the shrunk preview
        origin_x, origin_y = self._zoom_origin if self._zoom_origin is not None else (0, h - view_h)
        origin_x = (w - view_w) / 2 if view_w >= w else max(0, min(origin_x, w - view_w))
        origin_y = 0 if view_h >= h else max(0, min(origin_y, h - view_h))

        if self._advance is not None:
            elapsed = (wx.GetUTCTimeMillis() - self._advance["started"]) / ADVANCE_MS
            if elapsed >= 1 or abs(self._advance["base_scale"] - base_scale) > 1e-6:
                self._advance = None
                self._advance_timer.Stop()
            else:
                eased = 1 - (1 - elapsed) ** 3
                origin_y = self._advance["from"] + (origin_y - self._advance["from"]) * eased
        self._view = (base_scale, scale, origin_x, origin_y)

        src_box = (
            int(origin_x) if origin_x > 0 else 0,
            int(origin_y),
            min(w, math.ceil(origin_x + view_w)),
            min(h, math.ceil(origin_y + view_h)),
        )
        disp_x = stitch_left + round((src_box[0] - origin_x) * scale)
        disp_y = round((src_box[1] - origin_y) * scale)
        disp_w = max(1, round((src_box[2] - src_box[0]) * scale))
        disp_h = max(1, round((src_box[3] - src_box[1]) * scale))
        # Zoomed past 1:1 shows real pixels so seams can be judged
        resample = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.BOX
        shown = image.crop(src_box).resize((disp_w, disp_h), resample)

        def to_canvas(region):
            start = max(disp_y, min(disp_y + disp_h, round((region[0] - origin_y) * scale)))
            end = max(start, min(disp_y + disp_h, round((region[1] - origin_y) * scale)))
            return (start, end)

        regions = compute_semantic_regions(
            image_height=h,
            latest_slice=(self._overlay.get("latest_slice_start", 0), self._overlay.get("latest_slice_end", 0)),
            matched_region=(self._overlay.get("matched_region_start", 0), self._overlay.get("matched_region_end", 0)),
            probe_region=(self._overlay.get("overlap_visual_start", 0), self._overlay.get("overlap_visual_end", 0)),
            static_top=self._overlay.get("static_top", 0),
            static_bottom=self._overlay.get("static_bottom", 0),
        )
        has_footprint = regions["footprint"][1] > regions["footprint"][0]
        has_probe = regions["probe"][1] > regions["probe"][0]
        footprint = to_canvas(regions["footprint"])
        probe = to_canvas(regions["probe"])
        static_top_band = to_canvas(regions["static_top_band"])
        static_bottom_band = to_canvas(regions["static_bottom_band"])

        # Chart paper: fixed vertical rules, horizontal rules that belong to the scale so the grid feeds with the stitch
        canvas = Image.new("RGB", (paper_w, paper_h), PAPER)
        draw = ImageDraw.Draw(canvas)
        step = _scale_step(scale)
        minor = step / 5
        for x in range(GRID_COLUMN, paper_w, GRID_COLUMN):
            draw.line([(x, 0), (x, paper_h)], fill=GRID_MAJOR if (x // GRID_COLUMN) % 5 == 0 else GRID_MINOR)
        value = math.floor(origin_y / minor) * minor
        while (value - origin_y) * scale < paper_h:
            y = round((value - origin_y) * scale)
            major = abs(value / step - round(value / step)) < 1e-6
            draw.line([(self._scale_w, y), (paper_w, y)], fill=GRID_MAJOR if major else GRID_MINOR)
            value += minor
        draw.line([(self._scale_w, 0), (self._scale_w, paper_h)], fill=GRID_MAJOR)

        if has_footprint or has_probe:
            # History steps back: grey and a stop darker, so the newest panel is the only bright colour on the paper
            history = Image.blend(shown.convert("L").convert("RGB"), Image.new("RGB", shown.size, (150, 156, 150)), 0.22)
            canvas.paste(history, (disp_x, disp_y))
        else:
            canvas.paste(shown, (disp_x, disp_y))
        if footprint[1] > footprint[0]:
            canvas.paste(shown.crop((0, footprint[0] - disp_y, disp_w, footprint[1] - disp_y)), (disp_x, footprint[0]))

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        image_left, image_right = disp_x, disp_x + disp_w - 1
        if has_probe:
            if not self._render_success or self._overlay.get("match_status") == "candidate":
                self._draw_band(overlay_draw, probe, image_left, image_right, (*EVENT, 26), (*EVENT, 150))
            else:
                self._draw_band(overlay_draw, probe, image_left, image_right, (*PEN, 22), (*PEN, 140))
            if self._overlay.get("static_top", 0) > 0 or self._overlay.get("static_bottom", 0) > 0:
                self._draw_band(overlay_draw, static_top_band, image_left, image_right, (*GRID_MAJOR, 60), (*GRID_MAJOR, 200))
                self._draw_band(overlay_draw, static_bottom_band, image_left, image_right, (*GRID_MAJOR, 60), (*GRID_MAJOR, 200))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        if origin_y > 0:
            # The paper's top edge fades into the paper colour: the stitch continues above the view
            fade_height = min(36, paper_h)
            fade_left, fade_right = max(0, image_left), min(paper_w, image_right + 1)
            mask = Image.linear_gradient("L").rotate(180).resize((fade_right - fade_left, fade_height))
            canvas.paste(PAPER, (fade_left, 0, fade_right, fade_height), mask.point(lambda v: v * 200 // 255))

        self._last_hover_regions = {
            "footprint": footprint,
            "probe": probe,
            "static_top_band": static_top_band,
            "static_bottom_band": static_bottom_band,
        }

        wx_img = wx.Image(paper_w, paper_h)
        wx_img.SetData(canvas.tobytes())
        self._paper_bitmap = wx_img.ConvertToBitmap()
        self.paper.Refresh()

    def _draw_band(self, draw, band, left, right, fill, edge):
        band_top, band_bottom = band
        if band_bottom <= band_top:
            return
        draw.rectangle([(left, band_top), (right, band_bottom)], fill=fill)
        draw.line([(left, band_top), (right, band_top)], fill=edge, width=1)
        draw.line([(left, max(band_top, band_bottom - 1)), (right, max(band_top, band_bottom - 1))], fill=edge, width=1)

    def _on_paper_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self.paper)
        dc.SetBackground(wx.Brush(wx.Colour(*PAPER)))
        dc.Clear()
        if self._paper_bitmap is None:
            return
        dc.DrawBitmap(self._paper_bitmap, 0, 0)

        base_scale, scale, origin_x, origin_y = self._view
        paper_h = self.paper.GetClientSize().height
        gc = wx.GraphicsContext.Create(dc)
        gc.SetFont(self._scale_font, SCALE_INK)

        # Scale: numerals right-aligned against the paper rule, major and minor ticks hanging off it
        step = _scale_step(scale)
        header_h = gc.GetTextExtent("px")[1] + GAP * 2
        value = math.floor(origin_y / (step / 5)) * (step / 5)
        gc.SetPen(wx.Pen(SCALE_INK, 1))
        while (value - origin_y) * scale < paper_h:
            y = (value - origin_y) * scale
            major = abs(value / step - round(value / step)) < 1e-6
            tick = 7 if major else 3
            gc.StrokeLine(self._scale_w - tick, y, self._scale_w, y)
            if major and y > header_h:
                label = f"{round(value):,}"
                text_w, text_h = gc.GetTextExtent(label)
                gc.DrawText(label, self._scale_w - tick - GAP - text_w, y - text_h / 2)
            value += step / 5
        header = "px" if abs(self.preview_zoom - 1.0) < 0.001 else f"px {round(self.preview_zoom * 100)}%"
        gc.SetBrush(wx.Brush(wx.Colour(*PAPER)))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.DrawRectangle(0, 0, self._scale_w - 1, header_h)
        text_w, text_h = gc.GetTextExtent(header)
        gc.DrawText(header, self._scale_w - GAP - text_w, GAP)

        if self._paper_margin_left() > self._scale_w:
            self._paint_trace(gc, scale, origin_y, paper_h)

    def _paint_trace(self, gc, scale, origin_y, paper_h):
        # Pen trace: x is the match score from the 0.4 floor to a perfect 1.0, y is the seam it aligned on
        left = self._scale_w + 3
        width = TRACE_W - 6

        def point(entry):
            score = max(0.0, min(1.0, (entry["score"] - TRACE_FLOOR) / (1 - TRACE_FLOOR)))
            return left + score * width, (entry["y"] - origin_y) * scale

        merged = sorted((entry for entry in self._trace if entry["state"] == "merged"), key=lambda e: e["y"])
        if merged:
            ink = wx.Colour(*PEN)
            gc.SetPen(wx.Pen(ink, 2))
            path = gc.CreatePath()
            path.MoveToPoint(*point(merged[0]))
            for entry in merged[1:]:
                path.AddLineToPoint(*point(entry))
            gc.StrokePath(path)
            gc.SetBrush(wx.Brush(ink))
            gc.SetPen(wx.TRANSPARENT_PEN)
            x, y = point(merged[-1])
            gc.DrawEllipse(x - 3, y - 3, 6, 6)

        event_ink = wx.Colour(*EVENT)
        gc.SetPen(wx.Pen(event_ink, 2))
        for entry in self._trace:
            x, y = point(entry)
            if not -4 <= y <= paper_h + 4:
                continue
            if entry["state"] == "failed":
                gc.StrokeLine(left, y, left + width, y)
            elif entry["state"] == "undone":
                gc.StrokeLine(x - 3, y - 3, x + 3, y + 3)
                gc.StrokeLine(x - 3, y + 3, x + 3, y - 3)
