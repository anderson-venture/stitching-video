#!/usr/bin/env python3
"""
Simple Panoramic Video Stitcher
Uses template matching and correlation instead of feature detection for more reliable results.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


class SimplePanoramaStitcher:
    def __init__(self, left_video_path, right_video_path, output_path="simple_panorama.mp4"):
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        self.output_path = output_path
        
        # Video properties
        self.left_cap = None
        self.right_cap = None
        self.writer = None
        
        # Stitching parameters
        self.offset_x = None
        self.offset_y = None
        self.panorama_size = None
        
        # Debug mode
        self.debug = True
        
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
    
    def find_overlap_by_correlation(self, left_frame, right_frame):
        """Find overlap using template matching and correlation"""
        h, w = left_frame.shape[:2]
        
        # Convert to grayscale for correlation
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Try different overlap regions from the right side of left image
        best_correlation = -1
        best_offset_x = 0
        best_offset_y = 0
        
        # Search parameters
        search_width = w // 2  # Search in right half of left image
        template_width = w // 3  # Use 1/3 of image width as template
        
        print("  Searching for best overlap position...")
        
        # Extract template from left side of right image
        template = right_gray[:, :template_width]
        
        # Search in the right portion of left image
        search_region = left_gray[:, search_width:]
        
        # Perform template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.3:  # Reasonable correlation threshold
            best_offset_x = search_width + max_loc[0]
            best_offset_y = max_loc[1]
            best_correlation = max_val
            
            print(f"  Found overlap: offset=({best_offset_x}, {best_offset_y}), correlation={best_correlation:.3f}")
            
            # Save debug visualization
            if self.debug:
                self.save_debug_correlation(left_frame, right_frame, template, 
                                          best_offset_x, best_offset_y, template_width)
            
            return best_offset_x, best_offset_y, True
        else:
            print(f"  Low correlation found: {max_val:.3f}")
            return 0, 0, False
    
    def save_debug_correlation(self, left_frame, right_frame, template, offset_x, offset_y, template_width):
        """Save debug visualization of correlation matching"""
        Path("debug").mkdir(exist_ok=True)
        
        # Create visualization
        debug_img = left_frame.copy()
        
        # Draw template region on left image
        cv2.rectangle(debug_img, (offset_x, offset_y), 
                     (offset_x + template_width, offset_y + template.shape[0]), 
                     (0, 255, 0), 3)
        
        # Add text
        cv2.putText(debug_img, f"Template Match: ({offset_x}, {offset_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite("debug/correlation_match.jpg", debug_img)
        
        # Save side-by-side comparison
        h, w = left_frame.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = left_frame
        comparison[:, w:] = right_frame
        
        # Draw matching regions
        cv2.rectangle(comparison, (offset_x, offset_y), 
                     (offset_x + template_width, offset_y + template.shape[0]), 
                     (0, 255, 0), 2)
        cv2.rectangle(comparison, (w, offset_y), 
                     (w + template_width, offset_y + template.shape[0]), 
                     (0, 255, 0), 2)
        
        cv2.imwrite("debug/correlation_comparison.jpg", comparison)
        print("  Debug correlation images saved")
    
    def calculate_simple_panorama_size(self, left_shape, right_shape, offset_x, offset_y):
        """Calculate panorama size for simple horizontal stitching"""
        h_left, w_left = left_shape[:2]
        h_right, w_right = right_shape[:2]
        
        # Calculate panorama dimensions
        # Assume right image starts at offset_x from left image
        panorama_width = max(w_left, offset_x + w_right)
        panorama_height = max(h_left, h_right + abs(offset_y))
        
        # Adjust for negative offsets
        left_start_x = max(0, -offset_x)
        left_start_y = max(0, -offset_y)
        right_start_x = max(0, offset_x)
        right_start_y = max(0, offset_y)
        
        if offset_x < 0:
            panorama_width = max(w_right, abs(offset_x) + w_left)
        if offset_y < 0:
            panorama_height = max(h_right, abs(offset_y) + h_left)
        
        print(f"  Panorama size: {panorama_width}x{panorama_height}")
        print(f"  Left position: ({left_start_x}, {left_start_y})")
        print(f"  Right position: ({right_start_x}, {right_start_y})")
        
        return (panorama_width, panorama_height), (left_start_x, left_start_y), (right_start_x, right_start_y)
    
    def stitch_frame_simple(self, left_frame, right_frame, panorama_size, left_pos, right_pos):
        """Simple frame stitching with overlap blending"""
        panorama_width, panorama_height = panorama_size
        left_x, left_y = left_pos
        right_x, right_y = right_pos
        
        # Create panorama canvas
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Get frame dimensions
        h_left, w_left = left_frame.shape[:2]
        h_right, w_right = right_frame.shape[:2]
        
        # Place left frame
        left_end_x = min(left_x + w_left, panorama_width)
        left_end_y = min(left_y + h_left, panorama_height)
        left_w = left_end_x - left_x
        left_h = left_end_y - left_y
        
        panorama[left_y:left_end_y, left_x:left_end_x] = left_frame[:left_h, :left_w]
        
        # Place right frame
        right_end_x = min(right_x + w_right, panorama_width)
        right_end_y = min(right_y + h_right, panorama_height)
        right_w = right_end_x - right_x
        right_h = right_end_y - right_y
        
        # Check for overlap
        overlap_left = max(left_x, right_x)
        overlap_right = min(left_end_x, right_end_x)
        overlap_top = max(left_y, right_y)
        overlap_bottom = min(left_end_y, right_end_y)
        
        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            # There's overlap - blend it
            overlap_width = overlap_right - overlap_left
            overlap_height = overlap_bottom - overlap_top
            
            print(f"  Blending overlap region: {overlap_width}x{overlap_height}")
            
            # Extract overlapping regions
            left_overlap_x1 = overlap_left - left_x
            left_overlap_y1 = overlap_top - left_y
            left_overlap_x2 = left_overlap_x1 + overlap_width
            left_overlap_y2 = left_overlap_y1 + overlap_height
            
            right_overlap_x1 = overlap_left - right_x
            right_overlap_y1 = overlap_top - right_y
            right_overlap_x2 = right_overlap_x1 + overlap_width
            right_overlap_y2 = right_overlap_y1 + overlap_height
            
            # Blend the overlap region
            if (left_overlap_x1 >= 0 and left_overlap_y1 >= 0 and 
                left_overlap_x2 <= w_left and left_overlap_y2 <= h_left and
                right_overlap_x1 >= 0 and right_overlap_y1 >= 0 and 
                right_overlap_x2 <= w_right and right_overlap_y2 <= h_right):
                
                left_overlap = left_frame[left_overlap_y1:left_overlap_y2, 
                                        left_overlap_x1:left_overlap_x2]
                right_overlap = right_frame[right_overlap_y1:right_overlap_y2, 
                                          right_overlap_x1:right_overlap_x2]
                
                # Simple average blending
                blended = (left_overlap.astype(np.float32) + right_overlap.astype(np.float32)) / 2
                panorama[overlap_top:overlap_bottom, overlap_left:overlap_right] = blended.astype(np.uint8)
            
            # Add non-overlapping part of right frame
            if right_x < overlap_left:
                # Left part of right frame
                panorama[right_y:right_end_y, right_x:overlap_left] = \
                    right_frame[:right_h, :overlap_left-right_x]
            
            if overlap_right < right_end_x:
                # Right part of right frame
                start_col = overlap_right - right_x
                panorama[right_y:right_end_y, overlap_right:right_end_x] = \
                    right_frame[:right_h, start_col:start_col+(right_end_x-overlap_right)]
        else:
            # No overlap - just place right frame
            panorama[right_y:right_end_y, right_x:right_end_x] = right_frame[:right_h, :right_w]
        
        return panorama
    
    def initialize_videos(self):
        """Initialize video captures and find overlap"""
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
        
        # Try multiple frames to find consistent overlap
        test_frames = [30, 60, 100, 150, 200]  # Skip first few frames
        best_offset_x = 0
        best_offset_y = 0
        found_overlap = False
        
        for frame_idx in test_frames:
            print(f"\nTrying frame {frame_idx}...")
            
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret_left, left_frame = self.left_cap.read()
            ret_right, right_frame = self.right_cap.read()
            
            if not ret_left or not ret_right:
                continue
            
            offset_x, offset_y, success = self.find_overlap_by_correlation(left_frame, right_frame)
            
            if success:
                best_offset_x = offset_x
                best_offset_y = offset_y
                found_overlap = True
                print(f"  Using overlap from frame {frame_idx}")
                break
        
        if not found_overlap:
            # Fallback to simple side-by-side with estimated overlap
            print("  No reliable overlap found, using estimated overlap")
            best_offset_x = int(left_props['width'] * 0.7)  # Assume 30% overlap
            best_offset_y = 0
        
        self.offset_x = best_offset_x
        self.offset_y = best_offset_y
        
        # Calculate panorama dimensions using first frame
        self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_left, left_frame = self.left_cap.read()
        ret_right, right_frame = self.right_cap.read()
        
        self.panorama_size, self.left_pos, self.right_pos = self.calculate_simple_panorama_size(
            left_frame.shape, right_frame.shape, self.offset_x, self.offset_y)
        
        return left_props, right_props
    
    def stitch_videos(self):
        """Main stitching function"""
        try:
            left_props, right_props = self.initialize_videos()
            
            # Use the lower FPS for output
            output_fps = min(left_props['fps'], right_props['fps'])
            print(f"\nOutput FPS: {output_fps:.2f}")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, output_fps, 
                                        self.panorama_size)
            
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
                # Read frames
                self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, int(left_frame_pos))
                self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, int(right_frame_pos))
                
                ret_left, left_frame = self.left_cap.read()
                ret_right, right_frame = self.right_cap.read()
                
                if not ret_left or not ret_right:
                    print("Reached end of one or both videos")
                    break
                
                # Stitch frames
                panorama_frame = self.stitch_frame_simple(left_frame, right_frame, 
                                                        self.panorama_size, 
                                                        self.left_pos, self.right_pos)
                
                # Write frame
                self.writer.write(panorama_frame)
                
                # Update positions
                left_frame_pos += left_frame_interval
                right_frame_pos += right_frame_interval
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            print(f"\nSimple panoramic stitching completed! Output saved to: {self.output_path}")
            
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
    output_video = "simple_panorama_output.mp4"
    
    # Check if input files exist
    if not os.path.exists(left_video):
        print(f"Error: Left video not found: {left_video}")
        return 1
    
    if not os.path.exists(right_video):
        print(f"Error: Right video not found: {right_video}")
        return 1
    
    print("=== Simple Panoramic Video Stitching Tool ===")
    print(f"Left video: {left_video}")
    print(f"Right video: {right_video}")
    print(f"Output: {output_video}")
    print()
    
    # Create stitcher and process
    stitcher = SimplePanoramaStitcher(left_video, right_video, output_video)
    
    if stitcher.stitch_videos():
        print("Success! Simple panoramic video stitching completed.")
        print("Check the debug/ folder for correlation matching visualizations.")
        return 0
    else:
        print("Failed to stitch videos.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
