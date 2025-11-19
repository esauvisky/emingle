import wx
import numpy as np
from PIL import Image

class LivePreviewFrame(wx.Frame):
    def __init__(self, screen_height, debug_mode=False):
        # Create a tall, narrow window on the right side
        width = 400 if not debug_mode else 500
        height = 600
        style = wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW | wx.CAPTION | wx.RESIZE_BORDER
        super().__init__(None, title="Live Stitcher", size=(width, height), style=style)
        
        self.debug_mode = debug_mode

        # Position at top-right
        display_width, _ = wx.DisplaySize()
        self.SetPosition((display_width - width - 50, 50))

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
        self.Show()

    def update_image(self, pil_image, status="Merged", success=True, debug_info=None):
        """
        Updates the preview with the BOTTOM part of the huge merged image.
        """
        self.status_label.SetLabel(status)
        
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

        # 1. Crop the bottom X pixels to fit in the preview
        # We don't want to render a 20,000px image every frame.
        w, h = pil_image.size

        # Scale down to fit width
        target_w = self.GetClientSize().width - 10
        scale = target_w / w

        # We want to show the bottom portion of the image
        view_h_pixels = int(self.GetClientSize().height / scale)

        crop_top = max(0, h - view_h_pixels)
        crop = pil_image.crop((0, crop_top, w, h))

        # Resize for display
        disp_w = int(crop.width * scale)
        disp_h = int(crop.height * scale)

        img_resized = crop.resize((disp_w, disp_h), Image.Resampling.BOX)

        # Convert to WX Bitmap
        wx_img = wx.Image(img_resized.width, img_resized.height)
        wx_img.SetData(img_resized.convert("RGB").tobytes())
        bmp = wx_img.ConvertToBitmap()

        self.image_ctrl.SetBitmap(bmp)
        self.panel.Refresh()

        # Return color to black after a moment (Visual flash effect)
        wx.CallLater(500, lambda: self.panel.SetBackgroundColour(wx.BLACK))
        wx.CallLater(500, self.panel.Refresh)
