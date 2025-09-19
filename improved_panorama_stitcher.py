#!/usr/bin/env python3
"""
Improved Panoramic Video Stitcher
Enhanced version with better alignment and debugging capabilities.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


class ImprovedPanoramaStitcher:
    def __init__(self, left_video_path, right_video_path, output_path="improved_panorama.mp4"):
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
        self.offset = None
        
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
    
    def detect_and_match_features(self, img1, img2, max_features=2000):
        """Enhanced feature detection and matching"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Try SIFT first (better for this type of matching)
        try:
            detector = cv2.SIFT_create(nfeatures=max_features)
        except:
            # Fallback to ORB if SIFT not available
            detector = cv2.ORB_create(nfeatures=max_features)
        
        # Find keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"  Insufficient features: Left={len(kp1) if kp1 else 0}, Right={len(kp2) if kp2 else 0}")
            return None, None, []
        
        # Match features
        if len(des1[0]) == len(des2[0]):  # Same descriptor type
            # Use FLANN matcher for SIFT, BruteForce for ORB
            if des1.dtype == np.float32:
                # FLANN for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                matches = matcher.knnMatch(des1, des2, k=2)
            else:
                # BruteForce for ORB
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = matcher.knnMatch(des1, des2, k=2)
        else:
            return None, None, []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Slightly more permissive
                    good_matches.append(m)
        
        print(f"  Features: Left={len(kp1)}, Right={len(kp2)}, Matches={len(good_matches)}")
        
        return kp1, kp2, good_matches
    
    def calculate_homography_robust(self, left_frame, right_frame):
        """Calculate homography with multiple attempts and validation"""
        kp1, kp2, matches = self.detect_and_match_features(left_frame, right_frame)
        
        if kp1 is None or len(matches) < 20:
            print(f"  Insufficient matches for homography: {len(matches) if matches else 0}")
            return None
        
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Try different RANSAC parameters
        ransac_configs = [
            {'threshold': 3.0, 'confidence': 0.99, 'max_iters': 2000},
            {'threshold': 5.0, 'confidence': 0.95, 'max_iters': 1000},
            {'threshold': 8.0, 'confidence': 0.90, 'max_iters': 500}
        ]
        
        best_homography = None
        best_inliers = 0
        
        for config in ransac_configs:
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                ransacReprojThreshold=config['threshold'],
                confidence=config['confidence'],
                maxIters=config['max_iters']
            )
            
            if homography is not None and mask is not None:
                inliers = np.sum(mask)
                inlier_ratio = inliers / len(matches)
                
                print(f"  RANSAC (thresh={config['threshold']}): {inliers}/{len(matches)} inliers ({inlier_ratio:.2f})")
                
                if inliers > best_inliers and inlier_ratio > 0.1:  # At least 10% inliers
                    best_homography = homography
                    best_inliers = inliers
        
        if best_homography is not None:
            print(f"  Best homography: {best_inliers} inliers")
            
            # Save debug visualization if enabled
            if self.debug:
                self.save_debug_matches(left_frame, right_frame, kp1, kp2, matches, best_homography)
            
            return best_homography
        
        print("  Failed to find reliable homography")
        return None
    
    def save_debug_matches(self, img1, img2, kp1, kp2, matches, homography):
        """Save debug visualization of feature matches"""
        Path("debug").mkdir(exist_ok=True)
        
        # Draw matches
        matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("debug/feature_matches.jpg", matches_img)
        
        # Show transformation result
        h, w = img2.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, homography)
        
        # Draw the transformed corners on left image
        debug_img = img1.copy()
        pts = np.int32(transformed_corners).reshape(-1, 2)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
        cv2.imwrite("debug/transformation_preview.jpg", debug_img)
        
        print("  Debug images saved to debug/ folder")
    
    def calculate_panorama_size_improved(self, left_shape, right_shape, homography):
        """Improved panorama size calculation"""
        h_left, w_left = left_shape[:2]
        h_right, w_right = right_shape[:2]
        
        # Get corners of both images
        left_corners = np.float32([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]).reshape(-1, 1, 2)
        right_corners = np.float32([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]]).reshape(-1, 1, 2)
        
        # Transform right image corners to left image coordinate system
        right_transformed = cv2.perspectiveTransform(right_corners, homography)
        
        # Combine all corners
        all_corners = np.concatenate([left_corners, right_transformed], axis=0)
        
        # Find bounding box
        x_coords = all_corners[:, 0, 0]
        y_coords = all_corners[:, 0, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Add some padding
        padding = 50
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Calculate panorama size and offset
        panorama_width = int(max_x - min_x)
        panorama_height = int(max_y - min_y)
        offset_x = int(-min_x)
        offset_y = int(-min_y)
        
        print(f"  Bounding box: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
        print(f"  Panorama size: {panorama_width}x{panorama_height}")
        print(f"  Offset: ({offset_x}, {offset_y})")
        
        return (panorama_width, panorama_height), (offset_x, offset_y)
    
    def stitch_frame_improved(self, left_frame, right_frame, homography, panorama_size, offset):
        """Improved frame stitching with better blending"""
        panorama_width, panorama_height = panorama_size
        offset_x, offset_y = offset
        
        # Create panorama canvas
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Create transformation matrices
        left_transform = np.array([[1, 0, offset_x],
                                 [0, 1, offset_y],
                                 [0, 0, 1]], dtype=np.float32)
        
        right_transform = homography.copy()
        right_transform[0, 2] += offset_x
        right_transform[1, 2] += offset_y
        
        # Warp both images
        left_warped = cv2.warpPerspective(left_frame, left_transform,
                                        (panorama_width, panorama_height))
        right_warped = cv2.warpPerspective(right_frame, right_transform,
                                         (panorama_width, panorama_height))
        
        # Create masks for each image
        left_mask = (left_warped.sum(axis=2) > 0).astype(np.uint8)
        right_mask = (right_warped.sum(axis=2) > 0).astype(np.uint8)
        overlap_mask = left_mask & right_mask
        
        # Start with left image
        panorama = left_warped.copy()
        
        # Add right image where there's no left image
        no_left_mask = (left_mask == 0) & (right_mask > 0)
        panorama[no_left_mask] = right_warped[no_left_mask]
        
        # Blend in overlap region
        if np.any(overlap_mask):
            # Create distance-based blending weights
            left_dist = cv2.distanceTransform(left_mask, cv2.DIST_L2, 5)
            right_dist = cv2.distanceTransform(right_mask, cv2.DIST_L2, 5)
            
            # Normalize weights in overlap region
            total_dist = left_dist + right_dist
            overlap_coords = np.where(overlap_mask > 0)
            
            for y, x in zip(overlap_coords[0], overlap_coords[1]):
                if total_dist[y, x] > 0:
                    left_weight = left_dist[y, x] / total_dist[y, x]
                    right_weight = right_dist[y, x] / total_dist[y, x]
                    
                    panorama[y, x] = (left_weight * left_warped[y, x].astype(np.float32) + 
                                    right_weight * right_warped[y, x].astype(np.float32)).astype(np.uint8)
        
        return panorama
    
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
        
        # Try multiple frames to find the best homography
        best_homography = None
        test_frames = [0, 30, 60, 100, 150]  # Test different time points
        
        for frame_idx in test_frames:
            print(f"\nTrying frame {frame_idx}...")
            
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret_left, left_frame = self.left_cap.read()
            ret_right, right_frame = self.right_cap.read()
            
            if not ret_left or not ret_right:
                continue
            
            homography = self.calculate_homography_robust(left_frame, right_frame)
            
            if homography is not None:
                best_homography = homography
                print(f"  Using homography from frame {frame_idx}")
                break
        
        if best_homography is None:
            raise ValueError("Could not calculate homography from any test frame")
        
        self.homography = best_homography
        
        # Use the first frame for size calculation
        self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_left, left_frame = self.left_cap.read()
        ret_right, right_frame = self.right_cap.read()
        
        # Calculate panorama dimensions
        self.panorama_size, self.offset = self.calculate_panorama_size_improved(
            left_frame.shape, right_frame.shape, self.homography)
        
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
                panorama_frame = self.stitch_frame_improved(left_frame, right_frame, 
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
            
            print(f"\nImproved panoramic stitching completed! Output saved to: {self.output_path}")
            
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
    output_video = "improved_panorama_output.mp4"
    
    # Check if input files exist
    if not os.path.exists(left_video):
        print(f"Error: Left video not found: {left_video}")
        return 1
    
    if not os.path.exists(right_video):
        print(f"Error: Right video not found: {right_video}")
        return 1
    
    print("=== Improved Panoramic Video Stitching Tool ===")
    print(f"Left video: {left_video}")
    print(f"Right video: {right_video}")
    print(f"Output: {output_video}")
    print()
    
    # Create stitcher and process
    stitcher = ImprovedPanoramaStitcher(left_video, right_video, output_video)
    
    if stitcher.stitch_videos():
        print("Success! Improved panoramic video stitching completed.")
        print("Check the debug/ folder for feature matching visualizations.")
        return 0
    else:
        print("Failed to stitch videos.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
