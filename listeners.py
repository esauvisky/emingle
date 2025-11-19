import time
from threading import Thread
from loguru import logger
from pynput import keyboard, mouse

class KeyboardListener:
    def __init__(self):
        self.exit_event = False

    def start(self):
        listener_thread = Thread(target=self._listen_keyboard, daemon=True)
        listener_thread.start()

    def _listen_keyboard(self):
        with keyboard.Listener(on_press=self._on_press) as listener:
            listener.join()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.exit_event = True
                logger.info("Escape key pressed. Exiting...")
                return False
        except AttributeError:
            pass

class MouseScrollListener:
    def __init__(self, keyboard_listener):
        self.keyboard_listener = keyboard_listener
        # Store the exact time the last scroll happened
        self.last_scroll_time = 0

    def start(self):
        listener_thread = Thread(target=self._listen_mouse, daemon=True)
        listener_thread.start()

    def _listen_mouse(self):
        with mouse.Listener(on_scroll=self._on_scroll) as listener:
            listener.join()

    def _on_scroll(self, x, y, dx, dy):
        if self.keyboard_listener.exit_event:
            return False

        # Just update the timestamp. The main loop handles the logic.
        self.last_scroll_time = time.time()

class WxAppRunner:
    def __init__(self, app_class):
        self.app_class = app_class

    def start(self):
        app_thread = Thread(target=self._run_app, daemon=True)
        app_thread.start()
        self.app_thread = app_thread

    def _run_app(self):
        self.app_class.MainLoop()
