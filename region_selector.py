import sys
from fastapi import background
from loguru import logger

class RegionSelector:
    @staticmethod
    def select_region():
        return RegionSelector._select_region_wx()

    @staticmethod
    def _select_region_wx():
        try:
            import wx
        except ImportError:
            logger.error("wxPython is not installed. Please install wxPython using pip.")
            sys.exit(1)

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

                super().__init__(None, title="Select Region",  size=total_rect.GetSize(), style=wx.STAY_ON_TOP | wx.NO_BORDER | wx.TRANSPARENT_WINDOW | wx.BG_STYLE_TRANSPARENT )
                self.SetPosition(total_rect.GetPosition())
                self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
                self.SetBackgroundColour(wx.Colour(0,0,0,10))
                self.SetOwnBackgroundColour(wx.Colour(0,0,0,10))
                self.SetForegroundColour(wx.Colour(0,0,0,10))
                self.SetOwnForegroundColour(wx.Colour(0,0,0,10))
                self.SetTransparent(30)  # Adjust transparency level

                self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
                self.Bind(wx.EVT_MOTION, self.OnMouseMove)
                self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
                self.Bind(wx.EVT_PAINT, self.OnPaint)
                self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

                self.start_pos = None
                self.end_pos = None
                # self.SetSize(display_size)
                # self.SetPosition((0, 0))
                self.ShowFullScreen(True)

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
                if self.start_pos and self.end_pos:
                    dc = wx.PaintDC(self)
                    dc.SetPen(wx.Pen(wx.RED, 5))
                    dc.SetBrush(wx.RED_BRUSH)
                    rect = wx.Rect(self.start_pos, self.end_pos)
                    dc.DrawRectangle(rect)

            def OnEraseBackground(self, event):
                pass  # Prevent flickering

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

        return selection

if __name__ == '__main__':
    selector = RegionSelector()
    region = selector.select_region()
    print(f"Selected region: {region}")
