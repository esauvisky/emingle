import wx
import numpy as np
from PIL import Image

class LivePreviewFrame(wx.Frame):
    def __init__(self, screen_height, debug_mode=False, selection_region=None):
        # Create a tall, narrow window on the right side
        self.initial_width = 400 if not debug_mode else 500
        self.initial_height = 600
        self.max_height = screen_height - 100  # Leave some margin
        
        style = wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW | wx.CAPTION | wx.RESIZE_BORDER
        super().__init__(None, title="Live Stitcher", size=(self.initial_width, self.initial_height), style=style)

        self.debug_mode = debug_mode
        self.selection_region = selection_region

        # Position relative to selection region
        self._position_window()

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.BLACK)

        # UI Elements
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Status Text
        self.status_label = wx.StaticText(self.panel, label="Ready...")
        self.status_label.SetForegroundColour(wx.WHITE)
        self.sizer.Add(self.status_label, 0, wx.ALL | wx.EXPAND, 5)

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

            self.debug_panel.SetSizer(debug_sizer)
            self.sizer.Add(self.debug_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Image Display Area
        self.image_ctrl = wx.StaticBitmap(self.panel)
        self.sizer.Add(self.image_ctrl, 1, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(self.sizer)

        self.last_merged_image = None
        self.last_height_added = 0
        self.Show()

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
        self.status_label.SetLabel(status)

        # Store height added for overlay
        if debug_info and 'height_added' in debug_info:
            self.last_height_added = debug_info['height_added']

        # Update debug info if provided
        if self.debug_mode and debug_info:
            self.debug_height.SetLabel(f"Total Height: {debug_info['total_height']}px")
            self.debug_added.SetLabel(f"Height Added: {debug_info['height_added']}px")
            self.debug_processing.SetLabel(f"Processing Time: {debug_info['processing_time']:.3f}s")
            self.debug_debounce.SetLabel(f"Debounce Time: {debug_info['debounce_time']:.3f}s")

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
        
        # Get current client height (excluding debug panel and status)
        current_client_height = self.GetClientSize().height
        if self.debug_mode:
            current_client_height -= self.debug_panel.GetSize().height
        current_client_height -= self.status_label.GetSize().height + 20  # margins
        
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
        client_height -= self.status_label.GetSize().height + 20
        
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

        # 4. Add overlay for newly added pixels
        if success and self.last_height_added > 0:
            img_with_overlay = self._add_overlay(img_resized, crop_top, h, scale)
        else:
            img_with_overlay = img_resized

        # 5. Convert to WX Bitmap
        wx_img = wx.Image(img_with_overlay.width, img_with_overlay.height)
        wx_img.SetData(img_with_overlay.convert("RGB").tobytes())
        bmp = wx_img.ConvertToBitmap()

        self.image_ctrl.SetBitmap(bmp)
        self.panel.Refresh()

        # Return color to black after a moment (Visual flash effect)
        wx.CallLater(500, self.panel.Refresh)

    def _add_overlay(self, img_resized, crop_top, original_height, scale):
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
