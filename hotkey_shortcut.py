import os
import sys

import win32com.client
import wx

SHORTCUT_PATH = os.path.join(os.environ["APPDATA"], r"Microsoft\Windows\Start Menu\Programs\Emingle.lnk")
DEFAULT_HOTKEY = "CTRL+ALT+E"
VALID_KEYS = {ord(c): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
VALID_KEYS.update({wx.WXK_F1 + i: f"F{i + 1}" for i in range(12)})


def display_hotkey(hotkey):
    return "+".join(part.capitalize() for part in hotkey.split("+"))


class HotkeyDialog(wx.Dialog):
    def __init__(self):
        super().__init__(None, title="Emingle")
        self.hotkey = DEFAULT_HOTKEY

        label = wx.StaticText(self, label="Press the key combination that will start a capture.\n"
                                          "It must include Ctrl or Alt.")
        self.field = wx.TextCtrl(self, value=display_hotkey(self.hotkey), style=wx.TE_READONLY | wx.TE_CENTER)
        buttons = self.CreateButtonSizer(wx.OK | wx.CANCEL)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(label, 0, wx.ALL, 12)
        sizer.Add(self.field, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)
        sizer.Add(buttons, 0, wx.ALL | wx.ALIGN_RIGHT, 12)
        self.SetSizerAndFit(sizer)

        self.Bind(wx.EVT_CHAR_HOOK, self.OnCharHook)

    def OnCharHook(self, event):
        # Without Ctrl or Alt it can't be a shortcut hotkey, so let Enter, Escape and Tab do their usual thing
        if not (event.ControlDown() or event.AltDown()):
            event.Skip()
            return
        key = VALID_KEYS.get(event.GetKeyCode())
        if key is None:
            return
        parts = [name for name, down in (("CTRL", event.ControlDown()), ("ALT", event.AltDown()), ("SHIFT", event.ShiftDown())) if down]
        self.hotkey = "+".join(parts + [key])
        self.field.SetValue(display_hotkey(self.hotkey))


def setup_hotkey_shortcut():
    """Returns True when this launch should go on to capture."""
    shell = win32com.client.Dispatch("WScript.Shell")

    if os.path.exists(SHORTCUT_PATH):
        # Repoint it on every launch in case the exe was moved; the hotkey is kept
        shortcut = shell.CreateShortCut(SHORTCUT_PATH)
        shortcut.TargetPath = sys.executable
        shortcut.save()
        return True

    _app = wx.App(False)
    dialog = HotkeyDialog()
    if dialog.ShowModal() != wx.ID_OK:
        return False

    shortcut = shell.CreateShortCut(SHORTCUT_PATH)
    shortcut.TargetPath = sys.executable
    shortcut.Hotkey = dialog.hotkey
    shortcut.save()

    wx.MessageBox(f"Press {display_hotkey(dialog.hotkey)} anytime to capture a scrolling screenshot.\n\n"
                  "To change it, open the Start Menu, right-click Emingle > Open file location, "
                  "then edit the shortcut's Properties.", "Emingle is ready")
    return False
