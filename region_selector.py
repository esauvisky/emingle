
# Check Operating System
import os
import platform
import re
import shlex
import subprocess
import sys
import utils
from loguru import logger

OS_NAME = platform.system()

class RegionSelector:
    @staticmethod
    def select_region():
        if OS_NAME == 'Windows':
            return RegionSelector._select_region_windows()
        elif OS_NAME == 'Linux':
            # Verify if DISPLAY is set to ensure X11 session
            if os.environ.get('DISPLAY') is None:
                logger.error("DISPLAY environment variable not set. Ensure you are running an X11 session.")
                sys.exit(1)
            return RegionSelector._select_region_linux()
        else:
            logger.error(f"Unsupported Operating System: {OS_NAME}")
            sys.exit(1)

    @staticmethod
    def _select_region_windows():
        import tkinter as tk

        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)  # Transparency
        root.attributes('-topmost', True)
        root.config(cursor="cross")

        start_x = start_y = end_x = end_y = 0
        selection = None

        # Create a canvas to draw the selection rectangle
        canvas = tk.Canvas(root, cursor="cross", bg="grey")
        canvas.pack(fill=tk.BOTH, expand=True)

        def on_mouse_down(event):
            nonlocal start_x, start_y
            start_x, start_y = event.x, event.y
            canvas.delete("sel_rect")
            canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2, tags="sel_rect")

        def on_mouse_move(event):
            nonlocal start_x, start_y, end_x, end_y
            end_x, end_y = event.x, event.y
            canvas.coords("sel_rect", start_x, start_y, end_x, end_y)

        def on_mouse_up(event):
            nonlocal start_x, start_y, end_x, end_y, selection
            end_x, end_y = event.x, event.y
            root.destroy()
            x = min(start_x, end_x)
            y = min(start_y, end_y)
            w = abs(end_x - start_x)
            h = abs(end_y - start_y)
            selection = {'top': y, 'left': x, 'width': w, 'height': h}

        # Bind mouse events
        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

        root.mainloop()
        return selection

    @staticmethod
    def _select_region_linux():
        try:
            result = subprocess.run(shlex.split('import -format "%wx%h%X%Y\n" info:'), capture_output=True, text=True, check=True)
            geometry = result.stdout.strip()
            if not geometry:
                raise ValueError("No geometry received from import.")
            w, h, l, t = tuple(map(int, re.findall(r'\d+', geometry)))
            selection = {'left': l, 'top': t, 'width': w, 'height': h}
            return selection
        except FileNotFoundError:
            logger.error("`import` is not available. Please install imagemagick using your package manager (e.g., sudo apt-get install imagemagick).")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Unexpected error during region selection: {e}")
            sys.exit(1)
