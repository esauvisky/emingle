# Emingle

Automatically stitches screenshots as you scroll. Because manually taking 47 screenshots and aligning them in GIMP is apparently how some people spend their weekends.

## Demo

https://github.com/user-attachments/assets/b8f0b0d3-06f3-4bb7-9b8d-35dc314afa10

## Install & Run

```bash
pip install -r requirements.txt
python emingle.py
```

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
- Windows: `pip install pywin32`

## Debug Mode

`python emingle.py --debug` for when things inevitably break.
