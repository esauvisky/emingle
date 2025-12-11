import wx
import numpy as np
from PIL import Image

class LivePreviewFrame(wx.Frame):
    def __init__(self, screen_height, debug_mode=False, selection_region=None, manual_callback=None, undo_callback=None):
        # Create a tall, narrow window on the right side
        self.initial_width = 400 if not debug_mode else 500
        self.initial_height = 800  # Increased to fit new controls
        self.max_height = screen_height - 100  # Leave some margin

        style = wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW | wx.CAPTION | wx.RESIZE_BORDER
        super().__init__(None, title="Live Stitcher", size=(self.initial_width, self.initial_height), style=style)

        self.debug_mode = debug_mode
        self.selection_region = selection_region
        self.manual_callback = manual_callback
        self.undo_callback = undo_callback

        # Position relative to selection region
        self._position_window()

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
        self.sizer.Add(self.image_ctrl, 1, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(self.sizer)

        self.last_merged_image = None
        self.last_height_added = 0
        self.last_static_top = 0
        self.last_static_bottom = 0
        self.Show()

    def get_tolerance(self):
        """Get current tolerance value from slider"""
        return self.tolerance_slider.GetValue()

    def get_scroll_trigger_enabled(self):
        """Get current scroll trigger setting"""
        return self.scroll_trigger_checkbox.GetValue()

    def _position_window(self):
        """Position window relative to selection region"""
        display_width, display_height = wx.DisplaySize()

        if self.selection_region:
            sel_right = self.selection_region['left'] + self.selection_region['width']
            sel_top = self.selection_region['top']

            # Try to position to the top-right of selection
            pos_x = sel_right + 10
            pos_y = sel_top

            # If no space on the right, position to the top-left
            if pos_x + self.initial_width > display_width:
                pos_x = self.selection_region['left'] - self.initial_width - 10

            # Ensure we don't go off screen
            pos_x = max(0, min(pos_x, display_width - self.initial_width))
            pos_y = max(0, min(pos_y, display_height - self.initial_height))

            self.SetPosition((pos_x, pos_y))
        else:
            # Fallback to top-right corner
            self.SetPosition((display_width - self.initial_width - 50, 50))

    def update_image(self, pil_image, status="Merged", success=True, debug_info=None):
        """
        Updates the preview with the BOTTOM part of the huge merged image.
        """

        # Store overlay data
        if debug_info:
            if 'height_added' in debug_info:
                self.last_height_added = debug_info['height_added']
            if 'static_top' in debug_info:
                self.last_static_top = debug_info['static_top']
            if 'static_bottom' in debug_info:
                self.last_static_bottom = debug_info['static_bottom']

        # Update debug info if provided
        if self.debug_mode and debug_info:
            self.debug_height.SetLabel(f"Total Height: {debug_info['total_height']}px")
            self.debug_added.SetLabel(f"Height Added: {debug_info['height_added']}px")
            self.debug_processing.SetLabel(f"Processing Time: {debug_info['processing_time']:.3f}s")
            self.debug_debounce.SetLabel(f"Debounce Time: {debug_info['debounce_time']:.3f}s")
            self.debug_static_top.SetLabel(f"Static Top: {debug_info['static_top']}px")
            self.debug_static_bottom.SetLabel(f"Static Bottom: {debug_info['static_bottom']}px")

        if success:
            self.panel.SetBackgroundColour("#228B22") # Forest Green
        else:
            self.panel.SetBackgroundColour("#DC143C") # Crimson Red

        # 1. Calculate required display size
        w, h = pil_image.size
        target_w = self.GetClientSize().width - 10
        scale = target_w / w

        # Calculate how much height we need to show the full image
        required_display_height = int(h * scale)

        # Get current client height (excluding debug panel)
        current_client_height = self.GetClientSize().height
        if self.debug_mode:
            current_client_height -= self.debug_panel.GetSize().height
        current_client_height -= 20  # margins

        # Resize window if image overflows
        if required_display_height > current_client_height:
            new_window_height = min(required_display_height + 100, self.max_height)  # +100 for UI elements
            if self.debug_mode:
                new_window_height += self.debug_panel.GetSize().height

            current_size = self.GetSize()
            self.SetSize((current_size.width, new_window_height))

        # 2. Determine what portion of the image to show
        client_height = self.GetClientSize().height
        if self.debug_mode:
            client_height -= self.debug_panel.GetSize().height
        client_height -= 120  # Account for buttons and settings panel

        view_h_pixels = int(client_height / scale)

        # Show bottom portion if image is too tall
        if h > view_h_pixels:
            crop_top = h - view_h_pixels
            crop = pil_image.crop((0, crop_top, w, h))
        else:
            crop = pil_image
            crop_top = 0

        # 3. Resize for display
        disp_w = int(crop.width * scale)
        disp_h = int(crop.height * scale)
        img_resized = crop.resize((disp_w, disp_h), Image.Resampling.BOX)

        # 4. Add overlays for newly added pixels and static borders
        img_with_overlay = img_resized
        if success and self.last_height_added > 0:
            img_with_overlay = self._add_new_pixels_overlay(img_with_overlay, crop_top, h, scale)
        if success and (self.last_static_top > 0 or self.last_static_bottom > 0):
            img_with_overlay = self._add_static_borders_overlay(img_with_overlay, crop_top, h, scale)

        # 5. Convert to WX Bitmap
        wx_img = wx.Image(img_with_overlay.width, img_with_overlay.height)
        wx_img.SetData(img_with_overlay.convert("RGB").tobytes())
        bmp = wx_img.ConvertToBitmap()

        self.image_ctrl.SetBitmap(bmp)
        self.panel.Refresh()

        # Return color to black after a moment (Visual flash effect)
        wx.CallLater(500, self.panel.Refresh)

    def _add_new_pixels_overlay(self, img_resized, crop_top, original_height, scale):
        """Add a colored overlay to highlight the newly added pixels"""
        from PIL import Image, ImageDraw

        # Calculate where the new pixels are in the cropped/scaled image
        new_pixels_height_scaled = int(self.last_height_added * scale)

        # The new pixels are at the bottom of the original image
        new_pixels_start_original = original_height - self.last_height_added

        # Check if the new pixels are visible in our crop
        if new_pixels_start_original >= crop_top:
            # New pixels are visible
            new_pixels_start_in_crop = new_pixels_start_original - crop_top
            new_pixels_start_scaled = int(new_pixels_start_in_crop * scale)

            # Create overlay
            overlay = Image.new('RGBA', img_resized.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Draw semi-transparent rectangle over new pixels
            draw.rectangle([
                (0, new_pixels_start_scaled),
                (img_resized.width, img_resized.height)
            ], fill=(255, 255, 0, 80))  # Yellow with transparency

            # Composite overlay onto image
            img_rgba = img_resized.convert('RGBA')
            result = Image.alpha_composite(img_rgba, overlay)
            return result.convert('RGB')

        return img_resized

    def _add_static_borders_overlay(self, img_resized, crop_top, original_height, scale):
        """Add colored overlays to highlight static top/bottom borders"""
        from PIL import Image, ImageDraw

        # Create overlay
        overlay = Image.new('RGBA', img_resized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Static top border (red)
        if self.last_static_top > 0:
            # The static top is at the bottom of the merged image (where new content was added)
            # It corresponds to the top of the newly captured slice
            static_top_start_original = original_height - self.last_height_added
            static_top_end_original = static_top_start_original + self.last_static_top

            # Check if visible in our crop
            if static_top_start_original >= crop_top and static_top_end_original > crop_top:
                start_in_crop = max(0, static_top_start_original - crop_top)
                end_in_crop = min(img_resized.height / scale, static_top_end_original - crop_top)

                start_scaled = int(start_in_crop * scale)
                end_scaled = int(end_in_crop * scale)

                if end_scaled > start_scaled:
                    draw.rectangle([
                        (0, start_scaled),
                        (img_resized.width, end_scaled)
                    ], fill=(255, 0, 0, 60))  # Red with transparency

        # Static bottom border (blue)
        if self.last_static_bottom > 0:
            # The static bottom is at the very bottom of the newly added content
            static_bottom_start_original = original_height - self.last_static_bottom
            static_bottom_end_original = original_height

            # Check if visible in our crop
            if static_bottom_start_original >= crop_top and static_bottom_end_original > crop_top:
                start_in_crop = max(0, static_bottom_start_original - crop_top)
                end_in_crop = min(img_resized.height / scale, static_bottom_end_original - crop_top)

                start_scaled = int(start_in_crop * scale)
                end_scaled = int(end_in_crop * scale)

                if end_scaled > start_scaled:
                    draw.rectangle([
                        (0, start_scaled),
                        (img_resized.width, end_scaled)
                    ], fill=(0, 0, 255, 60))  # Blue with transparency

        # Composite overlay onto image
        img_rgba = img_resized.convert('RGBA')
        result = Image.alpha_composite(img_rgba, overlay)
        return result.convert('RGB')
