#!/usr/bin/env python3
from io import BytesIO
import sys
import time
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from loguru import logger

class ClipboardManager:
    @staticmethod
    def copy_image_to_clipboard(image):
        """
        Copy a PIL Image to the system clipboard.

        Args:
            image (PIL.Image.Image): The image to copy.
        """
        try:
            # Check if a QApplication instance already exists
            app = QApplication.instance()
            if app is None:
                # Create a new QApplication instance if none exists
                app = QApplication(sys.argv)
                logger.debug("Created new QApplication instance.")
            else:
                logger.debug("Using existing QApplication instance.")

            # Convert PIL Image to bytes in PNG format
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            png_data = buffer.getvalue()
            buffer.close()
            logger.debug("Converted PIL Image to PNG bytes.")

            # Load QPixmap from PNG data
            pixmap = QPixmap()
            if not pixmap.loadFromData(png_data, 'PNG'):
                raise ValueError("Failed to load QPixmap from PNG data.")
            logger.debug("Loaded QPixmap from PNG data.")

            # Set QPixmap to clipboard
            clipboard = app.clipboard()
            clipboard.setPixmap(pixmap)
            logger.info("Image copied to clipboard successfully.")

            # Start the event loop to keep the application alive
            # Set a timer to exit the application after a short delay
            def exit_app():
                app.quit()
                logger.debug("Exiting QApplication event loop.")

            # Use a QTimer to quit the application after 1 second
            timer = QTimer()
            timer.timeout.connect(exit_app)
            timer.setSingleShot(True)
            timer.start(1000)  # Time in milliseconds

            # Start the event loop
            app.exec_()

        except Exception as e:
            logger.exception(f"Failed to copy image to clipboard: {e}")

def main():
    # Configure logger
    logger.add(sys.stderr, level="DEBUG")

    # Create a simple red image with text for demonstration
    image = Image.new('RGB', (200, 100), color='red')
    draw = ImageDraw.Draw(image)
    draw.text((10, 40), "Hello, Clipboard!", fill='white')

    # Copy the image to the clipboard
    ClipboardManager.copy_image_to_clipboard(image)

    logger.info("Done.")

if __name__ == "__main__":
    main()
