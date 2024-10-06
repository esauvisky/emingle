# Screenshot Merger: Because Scrolling is Too Hard

Welcome to Screenshot Merger, the tool for those who find scrolling and taking multiple screenshots just too darn difficult. Why bother with all that manual labor when you can have a script do it for you?

## What is this?

It's a magical piece of software that takes screenshots as you scroll, then merges them together. Because apparently, we've reached a point in human evolution where even scrolling is too much work.

## Requirements

- A computer (shocking, I know)
- Python 3.x (because who doesn't have Python installed these days?)
- A functioning scroll wheel on your mouse (if you don't have this, maybe it's time to join the 21st century)
- The ability to press the Escape key (it's usually the one labeled "Esc" - you're welcome)

### Additional Requirements for Linux Users
- ImageMagick (for the `import` command)
- Either `copyq` or `xclip` (for clipboard operations)

### Additional Requirements for Windows Users
- pywin32 (for clipboard operations)

## How to Use

1. Clone this repository (if you don't know how to do this, maybe stick to crayons and paper)
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
   (Yes, you actually have to install things. Welcome to software development!)

   For Linux users:
   ```
   sudo apt-get install imagemagick copyq
   ```
   or
   ```
   sudo apt-get install imagemagick xclip
   ```
   (Choose either copyq or xclip, whichever tickles your fancy)

   For Windows users:
   ```
   pip install pywin32
   ```
   (Because apparently, Windows needs extra help to do simple things)

3. Run the script:
   ```
   python screenshot_merger.py
   ```
   (Pro tip: You can assign a shortcut to this command if opening a terminal is too much work for you)

4. Select the region you want to capture. Try not to strain yourself.

5. Scroll through your content. The script will take a screenshot every 5 scrolls. It's like magic, but with more Python.

6. When you're done, press the Escape key. Yes, that's it. One key. You can do it!

7. The merged image will be copied to your clipboard. Paste it wherever you want and bask in the glory of your "hard work".

## Debugging

If things go wrong (and let's face it, they probably will), run the script with the `--debug` flag:

```
python screenshot_merger.py --debug
```

This will show you what's happening behind the scenes. Try not to be too amazed by the technical wizardry.

## Final Thoughts

Congratulations! You've successfully automated a task that probably would have taken you less time to do manually. But hey, at least now you can brag about using a custom Python script to take screenshots. Your friends will be so impressed.

Remember, with great power comes great responsibility. Use this tool wisely, or don't. We're not your boss.

Happy screenshotting, you lazy genius!
