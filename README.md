# Emingle

Automatically stitches screenshots as you scroll. Because manually taking 47 screenshots and aligning them in GIMP is apparently how some people spend their weekends.

## Demo

https://github.com/user-attachments/assets/b8f0b0d3-06f3-4bb7-9b8d-35dc314afa10

## Download

Grab a binary from [Releases](../../releases/latest):

- Windows: `emingle.exe`, double-click it. `emingle-debug.exe` opens a console with logs.
- Linux (X11, glibc 2.35+): `emingle-linux`, then `chmod +x emingle-linux && ./emingle-linux`

## Run from source

```bash
pip install -r requirements.txt
python emingle.py
```

## Releasing

Push a `v*` tag (`git tag v1.0.0 && git push origin v1.0.0`). GitHub Actions builds both binaries and attaches them to a release.

## Usage

1. Select area to capture
2. Scroll normally, wait 0.5s between scrolls
3. Press Escape when done
4. Image is copied to clipboard

Green = good, red = scroll up a bit.

## Requirements

- Python 3.7+
- Working scroll wheel
- Linux: `sudo apt-get install copyq`

## Debug Mode

`python emingle.py --debug` (or `emingle-debug.exe --debug` / `./emingle-linux --debug`) for when things inevitably break.
