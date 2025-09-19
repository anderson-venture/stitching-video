#!/usr/bin/env python3
"""
Panoramic Video Stitcher
Stitches overlapping videos into a seamless panorama using feature matching and blending.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


class PanoramaVideoStitcher:
    def __init__(self, left_video_path, right_video_path, output_path="panorama_stitched.mp4"):
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        self.output_path = output_path
        
        # Video properties
        self.left_cap = None
        self.right_cap = None
        self.writer = None
        
        # Stitching parameters
        self.homography = None
        self.panorama_size = None
        self.blend_width = 100  # Width of blending region
        
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
    
    def detect_and_match_features(self, img1, img2):
        """Detect features and find matches between two images"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None, None, []
        
        # Match features using FLANN matcher for better performance
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return kp1, kp2, good_matches
    
    def calculate_homography(self, left_frame, right_frame):
        """Calculate homography transformation between frames"""
        kp1, kp2, matches = self.detect_and_match_features(left_frame, right_frame)
        
        if kp1 is None or len(matches) < 10:
            return None
        
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Calculate homography with RANSAC
        homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                            cv2.RANSAC, 
                                            ransacReprojThreshold=5.0,
                                            confidence=0.99)
        
        inliers = np.sum(mask) if mask is not None else 0
        print(f"  Homography calculated with {inliers}/{len(matches)} inliers")
        
        return homography if inliers > 10 else None
    
    def calculate_panorama_size(self, left_shape, right_shape, homography):
        """Calculate the size of the output panorama"""
        h, w = right_shape[:2]
        
        # Get corners of right image
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to left image coordinate system
        transformed_corners = cv2.perspectiveTransform(corners, homography)
        
        # Combine with left image corners
        left_corners = np.float32([[0, 0], [left_shape[1], 0], 
                                 [left_shape[1], left_shape[0]], [0, left_shape[0]]]).reshape(-1, 1, 2)
        
        all_corners = np.concatenate([left_corners, transformed_corners], axis=0)
        
        # Find bounding box
        x_coords = all_corners[:, 0, 0]
        y_coords = all_corners[:, 0, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Calculate panorama size and offset
        panorama_width = int(max_x - min_x)
        panorama_height = int(max_y - min_y)
        offset_x = int(-min_x) if min_x < 0 else 0
        offset_y = int(-min_y) if min_y < 0 else 0
        
        return (panorama_width, panorama_height), (offset_x, offset_y)
    
    def create_blend_mask(self, img1, img2, overlap_region):
        """Create a blending mask for seamless stitching"""
        h, w = img1.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Define overlap region (assuming horizontal overlap)
        overlap_start, overlap_end = overlap_region
        overlap_width = overlap_end - overlap_start
        
        if overlap_width > 0:
            # Create linear blend in overlap region
            blend_width = min(self.blend_width, overlap_width // 2)
            
            # Left side gets full weight until blend region
            mask[:, :overlap_start + blend_width] = 1.0
            
            # Blend region: linear transition
            for i in range(blend_width):
                weight = 1.0 - (i / blend_width)
                mask[:, overlap_start + blend_width + i] = weight
        else:
            mask[:, :] = 1.0
        
        return mask
    
    def stitch_frame(self, left_frame, right_frame, homography, panorama_size, offset):
        """Stitch two frames into a panorama"""
        panorama_width, panorama_height = panorama_size
        offset_x, offset_y = offset
        
        # Create panorama canvas
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Create transformation matrix for left image (with offset)
        left_transform = np.array([[1, 0, offset_x],
                                 [0, 1, offset_y],
                                 [0, 0, 1]], dtype=np.float32)
        
        # Transform right image
        right_transform = homography.copy()
        right_transform[0, 2] += offset_x
        right_transform[1, 2] += offset_y
        
        # Warp right image
        right_warped = cv2.warpPerspective(right_frame, right_transform, 
                                         (panorama_width, panorama_height))
        
        # Place left image
        left_warped = cv2.warpPerspective(left_frame, left_transform,
                                        (panorama_width, panorama_height))
        
        # Simple blending: take non-zero pixels from each image
        mask_left = (left_warped.sum(axis=2) > 0).astype(np.uint8)
        mask_right = (right_warped.sum(axis=2) > 0).astype(np.uint8)
        mask_overlap = mask_left & mask_right
        
        # Combine images
        panorama = left_warped.copy()
        panorama[mask_right > 0] = right_warped[mask_right > 0]
        
        # Blend overlap region
        if np.any(mask_overlap):
            overlap_coords = np.where(mask_overlap)
            for y, x in zip(overlap_coords[0], overlap_coords[1]):
                # Simple average blending in overlap region
                panorama[y, x] = (left_warped[y, x].astype(np.float32) + 
                                right_warped[y, x].astype(np.float32)) / 2
        
        return panorama.astype(np.uint8)
    
    def initialize_videos(self):
        """Initialize video captures and calculate homography"""
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
        
        # Read first frames to calculate homography
        self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        ret_left, left_frame = self.left_cap.read()
        ret_right, right_frame = self.right_cap.read()
        
        if not ret_left or not ret_right:
            raise ValueError("Cannot read initial frames")
        
        print("Calculating homography transformation...")
        self.homography = self.calculate_homography(left_frame, right_frame)
        
        if self.homography is None:
            raise ValueError("Could not calculate homography - insufficient feature matches")
        
        # Calculate panorama dimensions
        self.panorama_size, self.offset = self.calculate_panorama_size(
            left_frame.shape, right_frame.shape, self.homography)
        
        print(f"Panorama size: {self.panorama_size[0]}x{self.panorama_size[1]}")
        
        return left_props, right_props
    
    def stitch_videos(self):
        """Main stitching function"""
        try:
            left_props, right_props = self.initialize_videos()
            
            # Use the lower FPS for output
            output_fps = min(left_props['fps'], right_props['fps'])
            print(f"Output FPS: {output_fps:.2f}")
            
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
                panorama_frame = self.stitch_frame(left_frame, right_frame, 
                                                 self.homography, self.panorama_size, 
                                                 self.offset)
                
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
            
            print(f"Panoramic stitching completed! Output saved to: {self.output_path}")
            
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
    output_video = "panorama_output.mp4"
    
    # Check if input files exist
    if not os.path.exists(left_video):
        print(f"Error: Left video not found: {left_video}")
        return 1
    
    if not os.path.exists(right_video):
        print(f"Error: Right video not found: {right_video}")
        return 1
    
    print("=== Panoramic Video Stitching Tool ===")
    print(f"Left video: {left_video}")
    print(f"Right video: {right_video}")
    print(f"Output: {output_video}")
    print()
    
    # Create stitcher and process
    stitcher = PanoramaVideoStitcher(left_video, right_video, output_video)
    
    if stitcher.stitch_videos():
        print("Success! Panoramic video stitching completed.")
        return 0
    else:
        print("Failed to stitch videos.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
