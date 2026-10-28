import ctypes
import sys
import time
import tkinter as tk
import wx

from live_preview import ACCENT

SELECT_HINT = "Drag a box around the area to capture  (Esc to cancel)"


class RegionSelector:
    @staticmethod
    def select_region():
        if sys.platform == "win32":
            return RegionSelector._select_region_windows()

        selection = {}

        class SelectionFrame(wx.Frame):
            def __init__(self):
                display_count = wx.Display.GetCount()
                total_rect = wx.Rect()
                for d in range(display_count):
                    display = wx.Display(d)
                    rect = display.GetClientArea()
                    total_rect.Union(rect)
                # self.ShowFullScreen(False)

                super().__init__(None, title="Select Region",  size=total_rect.GetSize(), style=wx.STAY_ON_TOP | wx.NO_BORDER | wx.TRANSPARENT_WINDOW | wx.FRAME_NO_TASKBAR)
                self.SetPosition(total_rect.GetPosition())
                self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
                self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
                # Same dimming as the Windows overlay, dark enough for the hint to read
                self.SetTransparent(102)

                self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
                self.Bind(wx.EVT_MOTION, self.OnMouseMove)
                self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
                self.Bind(wx.EVT_PAINT, self.OnPaint)
                self.Bind(wx.EVT_CHAR_HOOK, self.OnCharHook)

                self.start_pos = None
                self.end_pos = None
                self.ShowFullScreen(True)
                self.SetFocus()

            def OnLeftDown(self, event):
                self.start_pos = event.GetPosition()
                self.end_pos = self.start_pos
                self.CaptureMouse()

            def OnMouseMove(self, event):
                if self.HasCapture():
                    self.end_pos = event.GetPosition()
                    self.Refresh()

            def OnLeftUp(self, event):
                if self.HasCapture():
                    self.ReleaseMouse()
                self.end_pos = event.GetPosition()
                self.CaptureSelection()
                self.Close()

            def OnPaint(self, event):
                # Repaint the whole frame each time so a shrinking drag leaves no trails
                dc = wx.AutoBufferedPaintDC(self)
                dc.SetBackground(wx.BLACK_BRUSH)
                dc.Clear()
                if self.start_pos is None:
                    dc.SetFont(wx.Font(wx.FontInfo(28).Bold()))
                    dc.SetTextForeground(wx.WHITE)
                    text_w, text_h = dc.GetTextExtent(SELECT_HINT)
                    width, height = self.GetClientSize()
                    dc.DrawText(SELECT_HINT, (width - text_w) // 2, (height - text_h) // 2)
                    return
                dc.SetPen(wx.Pen(wx.Colour(*ACCENT), 2))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRectangle(wx.Rect(self.start_pos, self.end_pos))

            def OnCharHook(self, event):
                if event.GetKeyCode() == wx.WXK_ESCAPE:
                    if self.HasCapture():
                        self.ReleaseMouse()
                    self.Close()
                    return
                event.Skip()

            def CaptureSelection(self):
                left = min(self.start_pos.x, self.end_pos.x)
                top = min(self.start_pos.y, self.end_pos.y)
                width = abs(self.start_pos.x - self.end_pos.x)
                height = abs(self.start_pos.y - self.end_pos.y)
                selection['left'] = left
                selection['top'] = top
                selection['width'] = width
                selection['height'] = height

        app = wx.App(False, useBestVisual=True)
        frame = SelectionFrame()
        frame.Show()
        app.MainLoop()

        time.sleep(0.5)
        return selection

    @staticmethod
    def _select_region_windows():
        # The wx overlay never shows up on Windows; Tk's alpha/topmost toplevel is the well-trodden way there
        user32 = ctypes.windll.user32
        # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN: the box spanning every monitor
        left, top, width, height = (user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))
        selection = {}
        drag = {}

        root = tk.Tk()
        root.overrideredirect(True)
        root.geometry(f"{width}x{height}+{left}+{top}")
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.4)

        canvas = tk.Canvas(root, bg="black", highlightthickness=0, cursor="cross")
        canvas.pack(fill="both", expand=True)
        hint = canvas.create_text(width // 2, height // 2, fill="white", font=("Segoe UI", 28, "bold"), text=SELECT_HINT)

        def on_press(event):
            canvas.delete(hint)
            drag["start"] = (event.x, event.y)
            drag["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#%02x%02x%02x" % ACCENT, width=2)

        def on_drag(event):
            canvas.coords(drag["rect"], *drag["start"], event.x, event.y)

        def on_release(event):
            start_x, start_y = drag["start"]
            # Canvas coordinates are relative to the virtual screen origin; mss wants absolute ones
            selection["left"] = left + min(start_x, event.x)
            selection["top"] = top + min(start_y, event.y)
            selection["width"] = abs(event.x - start_x)
            selection["height"] = abs(event.y - start_y)
            root.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", lambda event: root.destroy())
        root.focus_force()
        root.mainloop()

        time.sleep(0.5)
        return selection


    @staticmethod
    def highlight_region(region):
        """Veils everything outside `region` while capturing. Needs a running wx.App."""

        class HighlightFrame(wx.Frame):
            def __init__(self):
                self.screen = wx.Rect()
                for d in range(wx.Display.GetCount()):
                    self.screen.Union(wx.Display(d).GetGeometry())

                # TRANSPARENT_WINDOW makes it click-through on Windows; FRAME_SHAPED allows the hole over the region
                super().__init__(None, title="Highlight Region", pos=self.screen.GetPosition(), size=self.screen.GetSize(),
                                 style=wx.STAY_ON_TOP | wx.NO_BORDER | wx.FRAME_NO_TASKBAR | wx.FRAME_SHAPED | wx.TRANSPARENT_WINDOW)
                self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
                self.SetTransparent(120)
                # The region is in absolute screen coordinates; the frame's are relative to its own origin
                self.hole = wx.Rect(region["left"] - self.screen.x, region["top"] - self.screen.y, region["width"], region["height"])
                self.Bind(wx.EVT_PAINT, self.OnPaint)

            def OnPaint(self, event):
                dc = wx.AutoBufferedPaintDC(self)
                dc.SetBackground(wx.BLACK_BRUSH)
                dc.Clear()
                # Border on the veil right outside the hole, so it never ends up in a capture
                dc.SetPen(wx.Pen(wx.Colour(*ACCENT), 2))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRectangle(wx.Rect(self.hole).Inflate(1, 1))

        frame = HighlightFrame()
        frame.ShowWithoutActivating()
        shape = wx.Region(wx.Rect(wx.Point(0, 0), frame.screen.GetSize()))
        shape.Subtract(frame.hole)
        if not frame.SetShape(shape):
            raise RuntimeError("SetShape failed on the highlight overlay")
        return frame


if __name__ == '__main__':
    region = RegionSelector.select_region()
    print(f"Selected region: {region}")
    app = wx.App(False)
    RegionSelector.highlight_region(region)
    app.MainLoop()
