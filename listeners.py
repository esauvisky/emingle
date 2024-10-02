
from threading import Thread
from loguru import logger
from pynput import keyboard, mouse  # Modified import


class KeyboardListener:
    def __init__(self):
        self.exit_event = False

    def start(self):
        listener_thread = Thread(target=self._listen_keyboard, daemon=True)
        listener_thread.start()

    def _listen_keyboard(self):
        with keyboard.Listener(on_press=self._on_press) as listener:  # type: ignore
            listener.join()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.exit_event = True
                logger.info("Escape key pressed. Exiting...")
                return False  # Stop listener
        except AttributeError:
            pass

class MouseScrollListener:
    def __init__(self):
        self.screenshot_event = False
        self.exit_event = False
        self.scroll_count = 0
        self.scroll_threshold = 2  # Number of scrolls before triggering screenshot

    def start(self):
        listener_thread = Thread(target=self._listen_mouse, daemon=True)
        listener_thread.start()

    def _listen_mouse(self):
        with mouse.Listener(on_scroll=self._on_scroll) as listener:
            listener.join()

    def _on_scroll(self, x, y, dx, dy):
        self.scroll_count += 1
        logger.debug(f"Mouse scrolled: count={self.scroll_count}")
        if self.scroll_count >= self.scroll_threshold:
            self.screenshot_event = True
            self.scroll_count = 0
