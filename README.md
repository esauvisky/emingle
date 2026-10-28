# Emingle

Automatically stitches screenshots as you scroll. Because manually taking 47 screenshots and aligning them in GIMP is apparently how some people spend their weekends.

## Demo

https://github.com/user-attachments/assets/b8f0b0d3-06f3-4bb7-9b8d-35dc314afa10

## Download

[![Latest release](https://img.shields.io/github/v/release/esauvisky/emingle)](https://github.com/esauvisky/emingle/releases/latest)

- **Windows:** [emingle-windows.zip](https://github.com/esauvisky/emingle/releases/latest/download/emingle-windows.zip). Extract it somewhere permanent and run `emingle\emingle.exe`. The first time you open it, it asks for a keyboard shortcut (default `Ctrl+Alt+E`) and adds itself to the Start Menu; after that, press the shortcut to capture. If you move the folder, open `emingle.exe` once from the new place so the shortcut follows it. [emingle-debug-windows.zip](https://github.com/esauvisky/emingle/releases/latest/download/emingle-debug-windows.zip) opens a console with logs.
- **Linux** (X11, glibc 2.35+): [emingle-linux](https://github.com/esauvisky/emingle/releases/latest/download/emingle-linux), then `chmod +x emingle-linux && ./emingle-linux`

## Run from source

```bash
pip install -r requirements.txt
python emingle.py
```

## Releasing

Push a `v*` tag (`git tag v1.0.0 && git push origin v1.0.0`). GitHub Actions builds both binaries and attaches them to a release.

## Usage

1. Select area to capture. Everything else gets dimmed while capturing (untick "Dim outside the capture area" to turn it off)
2. Scroll normally, wait 0.5s between scrolls, or press **Auto Scroll** and keep your hands off the mouse
3. When you reach the end, scroll once more, or press **Done** (or Escape). Auto Scroll stops by itself at the bottom
4. Image is copied to clipboard

Green = good, red = scroll up a bit. Auto Scroll calibrates its step to about 40% of the area per screenshot and backs up on its own when a merge fails.

## Requirements

- Python 3.7+
- Working scroll wheel
- Linux: `sudo apt-get install copyq`

## Debug Mode

`python emingle.py --debug` (or `emingle-debug.exe --debug` / `./emingle-linux --debug`) for when things inevitably break.
