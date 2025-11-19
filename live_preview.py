import wx
import numpy as np
from PIL import Image

class LivePreviewFrame(wx.Frame):
    def __init__(self, screen_height):
        # Create a tall, narrow window on the right side
        width = 400
        height = 600
        style = wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW | wx.CAPTION | wx.RESIZE_BORDER
        super().__init__(None, title="Live Stitcher", size=(width, height), style=style)

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

        # Image Display Area
        self.image_ctrl = wx.StaticBitmap(self.panel)
        self.sizer.Add(self.image_ctrl, 1, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(self.sizer)

        self.last_merged_image = None
        self.Show()

    def update_image(self, pil_image, status="Merged", success=True):
        """
        Updates the preview with the BOTTOM part of the huge merged image.
        """
        self.status_label.SetLabel(status)

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
