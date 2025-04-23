
from io import BytesIO
import platform
import shlex
import shutil
import subprocess
import sys
from loguru import logger

OS_NAME = platform.system()

# Conditional imports based on OS
if OS_NAME == 'Windows':
    try:
        import win32clipboard  # type: ignore
    except ImportError:
        logger.error("pywin32 is not installed. Please install it using 'pip install pywin32'")
        sys.exit(1)

class ClipboardManager:
    @staticmethod
    def copy_image_to_clipboard(image):
        if OS_NAME == 'Windows':
            ClipboardManager._copy_image_to_clipboard_windows(image)
        elif OS_NAME == 'Linux':
            ClipboardManager._copy_image_to_clipboard_linux(image)
        else:
            logger.error("Unsupported OS for clipboard operations.")

    @staticmethod
    def _copy_image_to_clipboard_windows(image):
        try:
            output = BytesIO()
            image.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]  # BMP file header is 14 bytes
            output.close()

            win32clipboard.OpenClipboard()  # type: ignore
            win32clipboard.EmptyClipboard()  # type: ignore
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)  # type: ignore
            win32clipboard.CloseClipboard()  # type: ignore
        except Exception as e:
            logger.exception(f"Failed to copy image to clipboard: {e}")

    @staticmethod
    def _copy_image_to_clipboard_linux(image):
        # Save image to a temporary PNG file
        temp_file_name = "/tmp/screenshot_merger.png"
        with open(temp_file_name, "wb") as temp_file:
            image.save(temp_file)
        try:
            if shutil.which("copyq") is not None:
                # Use copyq to copy the image to the clipboard
                bash_cmd = f"copyq copyImageToClipboard {temp_file_name}"
                subprocess.run(shlex.split(bash_cmd), timeout=10)
            elif shutil.which("xclip") is not None:
                bash_cmd = f"xclip -selection clipboard -t image/png -i {temp_file_name}"
                subprocess.run(shlex.split(bash_cmd), timeout=10)
            else:
                raise FileNotFoundError("copyq or xclip is not installed. Please install it using your package manager (e.g., sudo apt-get install copyq or sudo apt-get install xclip).")
        except FileNotFoundError as e:
            logger.error(e)
        except Exception as e:
            logger.exception(f"Failed to copy image to clipboard: {e}")
