# Video Stitching Tool

A comprehensive Python tool for video stitching with two modes:
1. **Side-by-side stitching** (`main.py`) - Simple horizontal concatenation
2. **Panoramic stitching** (`panorama_stitcher.py`) - Advanced feature-based stitching for overlapping videos

## Features

### Common Features (Both Modes)
- **Frame Rate Synchronization**: Automatically handles videos with different frame rates
- **Aspect Ratio Preservation**: Maintains video proportions while resizing
- **Progress Tracking**: Shows real-time processing progress
- **Virtual Environment Support**: Clean dependency management

### Panoramic Mode Features
- **SIFT Feature Detection**: Finds matching points between overlapping videos
- **Homography Calculation**: Computes geometric transformation for alignment
- **RANSAC Outlier Removal**: Robust estimation ignoring mismatched features
- **Seamless Blending**: Smooth transitions in overlap regions
- **Automatic Panorama Sizing**: Calculates optimal output dimensions

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Setup Virtual Environment (Recommended)

```bash
python -m venv venv
.\venv\Scripts\activate.ps1  # Windows PowerShell
# or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Mode 1: Side-by-Side Stitching

For simple horizontal concatenation of two videos:

```bash
python main.py
```

Output: `output_stitched.mp4`

### Mode 2: Panoramic Stitching (For Overlapping Videos)

For videos with overlapping content that need feature-based alignment:

```bash
python panorama_stitcher.py
```

Output: `panorama_output.mp4`

### Analyze Video Overlap

To understand how your videos overlap:

```bash
python analyze_overlap.py
```

This creates an `analysis/` folder with visual comparisons and feature matching results.

### Video Properties Handled

The tool automatically:
- Detects different frame rates (e.g., 30 FPS vs 29.97 FPS)
- Resamples to the lower frame rate for smooth playback
- Resizes videos to match height while preserving aspect ratio
- Synchronizes video duration

## Output

- **Format**: MP4 (H.264)
- **Resolution**: Combined width × minimum height
- **Frame Rate**: Minimum of input frame rates
- **Duration**: Minimum of input durations

## Example

For the provided iPhone videos:
- Left: 1920×1080 @ 30 FPS
- Right: 1920×1080 @ 29.97 FPS
- Output: 3840×1080 @ 29.97 FPS

## Technical Details

The tool uses:
- **OpenCV** for video processing and frame manipulation
- **NumPy** for efficient array operations
- **Frame interpolation** for smooth rate conversion
- **Horizontal concatenation** for side-by-side stitching

## Troubleshooting

- Ensure both input videos exist in the correct paths
- Check that videos are in a supported format (MOV, MP4, AVI)
- Verify sufficient disk space for the output file
- Install all required dependencies from `requirements.txt`
