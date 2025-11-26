# Emingle: Intelligent Screenshot Stitching

Emingle is an intelligent screenshot stitching tool that automatically captures and merges screenshots as you scroll through content. It uses advanced computer vision techniques to seamlessly stitch images together, creating long continuous screenshots of web pages, documents, or any scrollable content.

https://github.com/user-attachments/assets/3dbd3d1c-adab-4df8-91d5-1b10f5e67d03

## Features

- **Automatic Screenshot Capture**: Detects when you stop scrolling and automatically captures screenshots
- **Intelligent Image Merging**: Uses Sobel edge detection and template matching to find optimal merge points
- **Live Preview Window**: Real-time preview of the merged image with smart positioning
- **Debug Mode**: Comprehensive debugging information and visualizations
- **Cross-Platform**: Works on Windows and Linux
- **Smart Debouncing**: Only captures when scrolling has settled (0.5s delay)
- **Robust Overlap Detection**: Handles dynamic content like blinking cursors and changing elements

## Requirements

- Python 3.7+
- A functioning scroll wheel or trackpad
- Ability to press the Escape key to finish capture

### Python Dependencies

Install with: `pip install -r requirements.txt`

- loguru (logging)
- matplotlib (debug visualizations)
- numpy (image processing)
- scipy (edge detection)
- scikit-image (template matching)
- pillow (image handling)
- mss (screenshot capture)
- wxPython (GUI)
- pynput (input monitoring)

### System Dependencies

**Linux users:**
```bash
sudo apt-get install copyq  # or xclip for clipboard support
```

**Windows users:**
```bash
pip install pywin32  # for clipboard operations
```

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python emingle.py
   ```

3. **Select capture region:**
   - Click and drag to select the area you want to capture
   - The selection will be highlighted

4. **Start scrolling:**
   - Scroll through your content normally
   - The live preview window will appear showing the merged result
   - Wait 0.5 seconds after each scroll for the capture to trigger

5. **Monitor progress:**
   - Green background = successful merge
   - Red background = merge failed (scroll up slightly)
   - Yellow overlay highlights newly added content

6. **Finish capture:**
   - Press **Escape** when done
   - The final merged image is automatically copied to your clipboard

## Debug Mode

Run with `--debug` flag for detailed information:

```bash
python emingle.py --debug
```

Debug mode provides:
- Processing statistics in the live preview
- Detailed merge visualizations saved to `debug_output/`
- Sobel edge detection analysis
- Template matching scores
- Pixel difference maps

## How It Works

1. **Edge Detection**: Uses Sobel filters to detect structural features in images
2. **Template Matching**: Finds the best overlap region between consecutive screenshots
3. **Robust Validation**: Ignores dynamic content (cursors, animations) when validating overlaps
4. **Smart Merging**: Uses hard cuts at overlap centers to avoid ghosting effects
5. **Live Preview**: Shows real-time results with automatic window resizing

## Troubleshooting

**Merge failures (red background):**
- Scroll up slightly to create more overlap
- Ensure content has visible structure (text, lines, etc.)
- Avoid areas with only solid colors

**No captures happening:**
- Wait a full 0.5 seconds after scrolling stops
- Check that the mouse cursor is over the selected region
- Verify scroll events are being detected

**Clipboard issues:**
- Linux: Install `copyq` or `xclip`
- Windows: Ensure `pywin32` is installed

## Technical Details

- **Debounce Time**: 0.5 seconds after scroll stops
- **Overlap Detection**: Sobel edge detection with template matching
- **Validation Threshold**: Ignores worst 20% of pixels to handle dynamic content
- **Merge Strategy**: Hard cut at overlap center to prevent ghosting

## License

Open source - use responsibly and enjoy your effortless screenshot stitching!
