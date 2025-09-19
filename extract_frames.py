#!/usr/bin/env python3
"""
Extract sample frames from videos for testing image stitching
"""

import cv2
import numpy as np
import os
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, num_frames=20, prefix="frame"):
    """Extract evenly spaced frames from a video"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Calculate frame indices to extract
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        # Evenly distribute frames across the video
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    print(f"  Extracting {len(frame_indices)} frames...")
    
    extracted_count = 0
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Could not read frame {frame_idx}")
            continue
        
        # Save frame
        filename = f"{prefix}_{i:03d}_f{frame_idx:05d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        success = cv2.imwrite(filepath, frame)
        if success:
            extracted_count += 1
            if (i + 1) % 5 == 0:  # Progress update every 5 frames
                print(f"  Saved {i+1}/{len(frame_indices)} frames")
        else:
            print(f"  Error: Could not save {filename}")
    
    cap.release()
    print(f"  Successfully extracted {extracted_count} frames to {output_dir}")
    return True


def main():
    """Main function"""
    
    # Input videos
    left_video = "source/1/iphone_left.mov"
    right_video = "source/1/iphone_right.mov"
    
    # Output directories
    left_output = "frames/left"
    right_output = "frames/right"
    
    print("=== Frame Extraction Tool ===")
    
    # Check if input files exist
    if not os.path.exists(left_video):
        print(f"Error: Left video not found: {left_video}")
        return 1
    
    if not os.path.exists(right_video):
        print(f"Error: Right video not found: {right_video}")
        return 1
    
    # Extract frames from left video
    print("\n--- Extracting frames from left video ---")
    success_left = extract_frames_from_video(left_video, left_output, 20, "left")
    
    # Extract frames from right video
    print("\n--- Extracting frames from right video ---")
    success_right = extract_frames_from_video(right_video, right_output, 20, "right")
    
    if success_left and success_right:
        print(f"\n✅ Success! Frames extracted to:")
        print(f"   Left frames:  {left_output}/")
        print(f"   Right frames: {right_output}/")
        print(f"\nYou can now work with individual images to perfect the stitching algorithm.")
        return 0
    else:
        print(f"\n❌ Failed to extract frames from one or both videos.")
        return 1


if __name__ == "__main__":
    exit(main())
