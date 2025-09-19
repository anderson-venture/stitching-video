#!/usr/bin/env python3
"""
Simple Video Stitching Tool
Stitches two videos side by side with frame rate synchronization.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


class VideoStitcher:
    def __init__(self, left_video_path, right_video_path, output_path="output_stitched.mp4"):
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        self.output_path = output_path
        
        # Video properties
        self.left_cap = None
        self.right_cap = None
        self.writer = None
        
    def get_video_properties(self, cap):
        """Get video properties"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'duration': duration
        }
    
    def initialize_videos(self):
        """Initialize video captures"""
        self.left_cap = cv2.VideoCapture(self.left_video_path)
        self.right_cap = cv2.VideoCapture(self.right_video_path)
        
        if not self.left_cap.isOpened():
            raise ValueError(f"Cannot open left video: {self.left_video_path}")
        if not self.right_cap.isOpened():
            raise ValueError(f"Cannot open right video: {self.right_video_path}")
        
        # Get properties
        left_props = self.get_video_properties(self.left_cap)
        right_props = self.get_video_properties(self.right_cap)
        
        print(f"Left video: {left_props['width']}x{left_props['height']} @ {left_props['fps']:.2f} FPS")
        print(f"Right video: {right_props['width']}x{right_props['height']} @ {right_props['fps']:.2f} FPS")
        
        return left_props, right_props
    
    def stitch_videos(self):
        """Main stitching function"""
        try:
            left_props, right_props = self.initialize_videos()
            
            # Use the lower FPS for output to ensure smooth playback
            output_fps = min(left_props['fps'], right_props['fps'])
            print(f"Output FPS: {output_fps:.2f}")
            
            # Calculate output dimensions (side by side)
            # Assume both videos have the same height, resize if needed
            output_height = min(left_props['height'], right_props['height'])
            
            # Calculate aspect ratio preserving widths
            left_ratio = output_height / left_props['height']
            right_ratio = output_height / right_props['height']
            
            left_width = int(left_props['width'] * left_ratio)
            right_width = int(right_props['width'] * right_ratio)
            
            output_width = left_width + right_width
            
            print(f"Output dimensions: {output_width}x{output_height}")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, output_fps, 
                                        (output_width, output_height))
            
            if not self.writer.isOpened():
                raise ValueError("Cannot initialize video writer")
            
            # Calculate frame intervals for resampling
            left_frame_interval = left_props['fps'] / output_fps
            right_frame_interval = right_props['fps'] / output_fps
            
            frame_count = 0
            left_frame_pos = 0
            right_frame_pos = 0
            
            # Calculate total frames to process
            max_duration = min(left_props['duration'], right_props['duration'])
            total_frames = int(max_duration * output_fps)
            
            print(f"Processing {total_frames} frames...")
            
            while frame_count < total_frames:
                # Read left frame
                self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, int(left_frame_pos))
                ret_left, left_frame = self.left_cap.read()
                
                # Read right frame
                self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, int(right_frame_pos))
                ret_right, right_frame = self.right_cap.read()
                
                if not ret_left or not ret_right:
                    print("Reached end of one or both videos")
                    break
                
                # Resize frames to match output height
                left_resized = cv2.resize(left_frame, (left_width, output_height))
                right_resized = cv2.resize(right_frame, (right_width, output_height))
                
                # Stitch frames horizontally
                stitched_frame = np.hstack((left_resized, right_resized))
                
                # Write frame
                self.writer.write(stitched_frame)
                
                # Update frame positions
                left_frame_pos += left_frame_interval
                right_frame_pos += right_frame_interval
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            print(f"Stitching completed! Output saved to: {self.output_path}")
            
        except Exception as e:
            print(f"Error during stitching: {str(e)}")
            return False
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


def main():
    """Main function"""
    # Default paths
    left_video = "source/1/iphone_left.mov"
    right_video = "source/1/iphone_right.mov"
    output_video = "output_stitched.mp4"
    
    # Check if input files exist
    if not os.path.exists(left_video):
        print(f"Error: Left video not found: {left_video}")
        return 1
    
    if not os.path.exists(right_video):
        print(f"Error: Right video not found: {right_video}")
        return 1
    
    print("=== Video Stitching Tool ===")
    print(f"Left video: {left_video}")
    print(f"Right video: {right_video}")
    print(f"Output: {output_video}")
    print()
    
    # Create stitcher and process
    stitcher = VideoStitcher(left_video, right_video, output_video)
    
    if stitcher.stitch_videos():
        print("Success! Video stitching completed.")
        return 0
    else:
        print("Failed to stitch videos.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
