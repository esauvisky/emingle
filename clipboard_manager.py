#!/usr/bin/env python3
from io import BytesIO
import time
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QImage
import sys
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

            clipboard = app.clipboard()

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
            clipboard.setPixmap(pixmap)
            logger.info("Image copied to clipboard successfully.")

            # Process events to ensure the clipboard is updated
            app.processEvents()
            time.sleep(0.1);
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
