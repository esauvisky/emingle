#!/usr/bin/env python3
from loguru import logger
import tkinter as tk
from PIL import Image, ImageTk

Config  = {
    "DEBUG_MODE": False
}

def setup_logging(log_lvl="DEBUG", options={}):
    file = options.get("file", False)
    function = options.get("function", False)
    process = options.get("process", False)
    thread = options.get("thread", False)

    log_fmt = (u"<n><d><level>{time:HH:mm:ss.SSS} | " +
               f"{'{file:>15.15}' if file else ''}" +
               f"{'{function:>15.15}' if function else ''}" +
               f"{':{line:<4} | ' if file or function else ''}" +
               f"{'{process.name:>12.12} | ' if process else ''}" +
               f"{'{thread.name:<11.11} | ' if thread else ''}" +
               u"{level:1.1} | </level></d></n><level>{message}</level>")

    logger.configure(
        handlers=[{
            "sink": lambda x: print(x, end=""),
            "level": log_lvl,
            "format": log_fmt,
            "colorize": True,
            "backtrace": True,
            "diagnose": True
        }],
        levels=[
            {"name": "TRACE", "color": "<white><dim>"},
            {"name": "DEBUG", "color": "<cyan><dim>"},
            {"name": "INFO", "color": "<white>"}
        ]
    )  # type: ignore # yapf: disable




# Function to display images in debug mode
def display_images(new_image, merged_image):
    root = tk.Tk(sync=False)
    root.title("Debug Images")

    # Create frames for layout
    frame_new = tk.Frame(root)
    frame_new.pack(side="left", padx=10, pady=10)

    frame_merged = tk.Frame(root)
    frame_merged.pack(side="right", padx=10, pady=10)

    # New Captured Image
    new_image_tk = ImageTk.PhotoImage(new_image)
    new_image_label = tk.Label(frame_new, image=new_image_tk) # type: ignore
    new_image_label.pack()
    new_image_title = tk.Label(frame_new, text="New Captured Image")
    new_image_title.pack()

    # Merged Image
    if merged_image is not None:
        merged_image_tk = ImageTk.PhotoImage(merged_image)
        merged_image_label = tk.Label(frame_merged, image=merged_image_tk) # type: ignore
        merged_image_label.pack()
        merged_image_title = tk.Label(frame_merged, text="Current Merged Image")
        merged_image_title.pack()

    # Keep a reference to the images to prevent garbage collection
    root.mainloop()
