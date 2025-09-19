#!/usr/bin/env python3
"""
Simple Image Stitching Tool
Test stitching algorithms on individual image pairs before applying to video.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path


class ImageStitcher:
    def __init__(self, debug=True):
        self.debug = debug
        
    def load_image_pair(self, left_path, right_path):
        """Load a pair of images"""
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None:
            raise ValueError(f"Cannot load left image: {left_path}")
        if right_img is None:
            raise ValueError(f"Cannot load right image: {right_path}")
            
        return left_img, right_img
    
    def method1_simple_overlap(self, left_img, right_img, overlap_percent=0.3):
        """Method 1: Simple overlap assumption - assume right image overlaps left by X%"""
        h, w = left_img.shape[:2]
        
        # Calculate overlap
        overlap_width = int(w * overlap_percent)
        
        # Create panorama
        panorama_width = w + w - overlap_width
        panorama_height = h
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Place left image
        panorama[:, :w] = left_img
        
        # Place right image with overlap
        right_start = w - overlap_width
        
        # Simple blending in overlap region
        for x in range(overlap_width):
            alpha = x / overlap_width  # 0 to 1
            blend_col = right_start + x
            panorama[:, blend_col] = (
                (1 - alpha) * left_img[:, w - overlap_width + x].astype(np.float32) +
                alpha * right_img[:, x].astype(np.float32)
            ).astype(np.uint8)
        
        # Add non-overlapping part of right image
        panorama[:, w:] = right_img[:, overlap_width:]
        
        return panorama, f"simple_overlap_{overlap_percent:.1f}"
    
    def method2_template_matching(self, left_img, right_img):
        """Method 2: Use template matching to find overlap"""
        h, w = left_img.shape[:2]
        
        # Convert to grayscale for matching
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Use left portion of right image as template
        template_width = w // 3
        template = right_gray[:, :template_width]
        
        # Search in right portion of left image
        search_start = w // 2
        search_region = left_gray[:, search_start:]
        
        # Template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.3:
            print(f"  Warning: Low correlation {max_val:.3f}")
        
        # Calculate overlap position
        match_x = search_start + max_loc[0]
        match_y = max_loc[1]
        
        print(f"  Template match: position=({match_x}, {match_y}), correlation={max_val:.3f}")
        
        # Create panorama
        panorama_width = max(w, match_x + w)
        panorama_height = max(h, match_y + h)
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Place left image
        panorama[:h, :w] = left_img
        
        # Calculate overlap region
        overlap_left = match_x
        overlap_right = min(w, match_x + w)
        overlap_width = overlap_right - overlap_left
        
        if overlap_width > 0:
            # Blend overlap region
            for x in range(overlap_width):
                alpha = x / overlap_width
                blend_col = overlap_left + x
                right_col = x
                
                if blend_col < w and right_col < w:
                    panorama[match_y:match_y+h, blend_col] = (
                        (1 - alpha) * left_img[:, blend_col].astype(np.float32) +
                        alpha * right_img[:, right_col].astype(np.float32)
                    ).astype(np.uint8)
        
        # Add non-overlapping part of right image
        if overlap_right < match_x + w:
            right_start_col = overlap_width
            pano_start_col = overlap_right
            pano_end_col = min(panorama_width, match_x + w)
            right_end_col = right_start_col + (pano_end_col - pano_start_col)
            
            panorama[match_y:match_y+h, pano_start_col:pano_end_col] = \
                right_img[:, right_start_col:right_end_col]
        
        return panorama, f"template_match_{max_val:.3f}"
    
    def method3_manual_offset(self, left_img, right_img, offset_x, offset_y=0):
        """Method 3: Manual offset specification"""
        h, w = left_img.shape[:2]
        
        # Create panorama
        panorama_width = max(w, offset_x + w)
        panorama_height = max(h, abs(offset_y) + h)
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        
        # Calculate positions
        left_y = max(0, -offset_y)
        right_y = max(0, offset_y)
        
        # Place left image
        panorama[left_y:left_y+h, :w] = left_img
        
        # Calculate overlap
        overlap_left = max(0, offset_x)
        overlap_right = min(w, offset_x + w)
        overlap_width = overlap_right - overlap_left
        
        if overlap_width > 0:
            # Blend overlap
            for x in range(overlap_width):
                alpha = x / overlap_width
                left_col = overlap_left + x
                right_col = x
                blend_col = overlap_left + x
                
                panorama[right_y:right_y+h, blend_col] = (
                    (1 - alpha) * left_img[left_y:left_y+h, left_col].astype(np.float32) +
                    alpha * right_img[:, right_col].astype(np.float32)
                ).astype(np.uint8)
        
        # Add non-overlapping part of right image
        if offset_x + w > overlap_right:
            right_start = overlap_width
            pano_start = overlap_right
            panorama[right_y:right_y+h, pano_start:offset_x+w] = \
                right_img[:, right_start:w-(offset_x+w-pano_start)]
        
        return panorama, f"manual_offset_{offset_x}_{offset_y}"
    
    def stitch_image_pair(self, left_path, right_path, output_dir="stitched_images"):
        """Stitch a pair of images using multiple methods"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load images
        print(f"\nStitching: {os.path.basename(left_path)} + {os.path.basename(right_path)}")
        left_img, right_img = self.load_image_pair(left_path, right_path)
        
        # Get base name for output files
        left_base = os.path.splitext(os.path.basename(left_path))[0]
        right_base = os.path.splitext(os.path.basename(right_path))[0]
        output_base = f"{left_base}+{right_base}"
        
        results = []
        
        # Method 1: Simple overlap assumptions
        for overlap in [0.2, 0.3, 0.4, 0.5]:
            try:
                panorama, method_name = self.method1_simple_overlap(left_img, right_img, overlap)
                output_path = os.path.join(output_dir, f"{output_base}_{method_name}.jpg")
                cv2.imwrite(output_path, panorama)
                results.append((method_name, output_path, panorama.shape))
                print(f"  ✓ {method_name}: {panorama.shape[1]}x{panorama.shape[0]}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
        
        # Method 2: Template matching
        try:
            panorama, method_name = self.method2_template_matching(left_img, right_img)
            output_path = os.path.join(output_dir, f"{output_base}_{method_name}.jpg")
            cv2.imwrite(output_path, panorama)
            results.append((method_name, output_path, panorama.shape))
            print(f"  ✓ {method_name}: {panorama.shape[1]}x{panorama.shape[0]}")
        except Exception as e:
            print(f"  ✗ template_match: {e}")
        
        # Method 3: Manual offsets
        manual_offsets = [
            (1000, 0),   # Assume 1000px overlap
            (1200, 0),   # Assume 1200px overlap
            (800, 0),    # Assume 800px overlap
            (1000, 20),  # With slight vertical offset
            (1000, -20), # With slight vertical offset
        ]
        
        for offset_x, offset_y in manual_offsets:
            try:
                panorama, method_name = self.method3_manual_offset(left_img, right_img, offset_x, offset_y)
                output_path = os.path.join(output_dir, f"{output_base}_{method_name}.jpg")
                cv2.imwrite(output_path, panorama)
                results.append((method_name, output_path, panorama.shape))
                print(f"  ✓ {method_name}: {panorama.shape[1]}x{panorama.shape[0]}")
            except Exception as e:
                print(f"  ✗ manual_{offset_x}_{offset_y}: {e}")
        
        return results


def main():
    """Main function - test multiple image pairs"""
    
    # Find available frames
    left_frames = sorted(glob.glob("frames/left/*.jpg"))
    right_frames = sorted(glob.glob("frames/right/*.jpg"))
    
    if not left_frames or not right_frames:
        print("Error: No frames found. Run extract_frames.py first.")
        return 1
    
    print("=== Image Stitching Test Tool ===")
    print(f"Found {len(left_frames)} left frames and {len(right_frames)} right frames")
    
    # Create stitcher
    stitcher = ImageStitcher(debug=True)
    
    # Test a few representative pairs
    test_pairs = [
        (0, "First frames"),
        (5, "Middle-early frames"),
        (10, "Middle frames"),
        (15, "Middle-late frames"),
        (19, "Last frames")
    ]
    
    all_results = []
    
    for idx, description in test_pairs:
        if idx < len(left_frames) and idx < len(right_frames):
            print(f"\n--- Testing {description} (pair {idx}) ---")
            
            results = stitcher.stitch_image_pair(left_frames[idx], right_frames[idx])
            all_results.extend(results)
        else:
            print(f"Skipping pair {idx} - not enough frames")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Generated {len(all_results)} stitched images in stitched_images/")
    print(f"Methods tested:")
    
    methods = {}
    for method_name, path, shape in all_results:
        method_type = method_name.split('_')[0] + "_" + method_name.split('_')[1]
        if method_type not in methods:
            methods[method_type] = 0
        methods[method_type] += 1
    
    for method, count in methods.items():
        print(f"  - {method}: {count} images")
    
    print(f"\nNow check the stitched_images/ folder to see which method works best!")
    print(f"Look for:")
    print(f"  - Proper alignment without doubling")
    print(f"  - Smooth blending in overlap regions") 
    print(f"  - No missing parts of the scene")
    
    return 0


if __name__ == "__main__":
    exit(main())
