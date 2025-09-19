# Video Stitching Tool

A Python tool for stitching two videos together using OpenCV and computer vision techniques.

## Features

- **Flexible Input Methods**: Multiple ways to specify input videos
- **Automatic Video Detection**: Auto-finds videos in source directory
- **Command Line Interface**: Full CLI support with arguments
- **Interactive Mode**: Prompts for input when needed
- **Video Resampling**: Ensures both videos have the same FPS
- **Feature Matching**: Uses ORB features for robust stitching

## Installation

1. Install required packages:
```bash
pip install -r requirement.txt
```

2. Ensure FFmpeg is installed on your system for video processing.

## Usage

### Method 1: Command Line Arguments (Recommended)

```bash
# Basic usage with auto-detection
python main.py

# Specify video files directly
python main.py --video1 path/to/video1.mov --video2 path/to/video2.mov

# Specify output file and FPS
python main.py -v1 video1.mov -v2 video2.mov -o output.mp4 --fps 24

# Use different source directory
python main.py --source-dir path/to/videos
```

### Method 2: Interactive Mode

If no command line arguments are provided, the tool will:
1. First try to auto-detect videos in the `source/1` directory
2. If no videos found, prompt you to enter file paths manually

### Method 3: Auto-Detection

The tool automatically looks for video files in the `source/1` directory and uses the first two found files.

## Command Line Options

- `--video1`, `-v1`: Path to first video file
- `--video2`, `-v2`: Path to second video file  
- `--output`, `-o`: Output video file path (default: stitched_video.mp4)
- `--fps`: Output FPS (default: 30)
- `--source-dir`, `-s`: Source directory containing video files (default: source/1)

## Supported Video Formats

- .mov
- .mp4
- .avi
- .mkv
- .wmv

## Example

```bash
# Using the default source directory
python main.py

# Output:
# Video Stitching Tool
# ==================================================
# Auto-detected videos: source/1/iphone_left.mov and source/1/iphone_right.mov
# 
# Processing videos:
#   Video 1: source/1/iphone_left.mov
#   Video 2: source/1/iphone_right.mov
#   Output: stitched_video.mp4
#   FPS: 30
```

## Notes

- The tool uses the first frame to compute the homography matrix for stitching
- Press 'q' during processing to quit early
- Temporary resampled videos are created during processing
- Make sure both input videos have sufficient overlap for good stitching results
