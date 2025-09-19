import cv2
import numpy as np
import subprocess
import os
import sys
import argparse
from pathlib import Path

def resample_video(input_path, output_path, fps=30):
    """
    Resample video to target FPS using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-r", str(fps),
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stitch_images(img1, img2, H=None):
    """
    Stitch two images using homography (H).
    If H is None, compute it using ORB feature matching.
    """
    if H is None:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:300]

        # Extract points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp img1
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]
    panorama = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    panorama[0:h2, 0:w2] = img2

    return panorama, H


def stitch_videos(video1_path, video2_path, output_path="stitched_output.mp4", fps=30):
    """
    Resample two videos, stitch them frame by frame, and save output.
    """
    # Resample both videos
    v1_fixed = "fixed_" + os.path.basename(video1_path)
    v2_fixed = "fixed_" + os.path.basename(video2_path)

    print(f"Resampling {video1_path} -> {v1_fixed} at {fps} fps...")
    resample_video(video1_path, v1_fixed, fps)

    print(f"Resampling {video2_path} -> {v2_fixed} at {fps} fps...")
    resample_video(video2_path, v2_fixed, fps)

    # Open videos
    cap1 = cv2.VideoCapture(v1_fixed)
    cap2 = cv2.VideoCapture(v2_fixed)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output writer
    out_width = width1 + width2
    out_height = max(height1, height2)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use 'avc1' for .mov
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    H = None  # homography placeholder

    print("Processing frames...")
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # First frame: compute homography
        if H is None:
            stitched, H = stitch_images(frame1, frame2, H)
        else:
            stitched, _ = stitch_images(frame1, frame2, H)

        out.write(stitched)

        # (Optional) show preview
        cv2.imshow("Stitched Video", stitched)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Stitched video saved to {output_path}")


def get_video_paths():
    """
    Get video file paths from command line arguments or user input.
    """
    parser = argparse.ArgumentParser(description='Stitch two videos together')
    parser.add_argument('--video1', '-v1', type=str, help='Path to first video file')
    parser.add_argument('--video2', '-v2', type=str, help='Path to second video file')
    parser.add_argument('--output', '-o', type=str, default='stitched_video.mp4', 
                       help='Output video file path (default: stitched_video.mp4)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Output FPS (default: 30)')
    parser.add_argument('--source-dir', '-s', type=str, default='source/1',
                       help='Source directory containing video files (default: source/1)')
    
    args = parser.parse_args()
    
    # If video paths not provided via command line, try to find them in source directory
    if not args.video1 or not args.video2:
        source_dir = Path(args.source_dir)
        if source_dir.exists():
            # Look for common video file patterns
            video_extensions = ['.mov', '.mp4', '.avi', '.mkv', '.wmv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(list(source_dir.glob(f'*{ext}')))
            
            if len(video_files) >= 2:
                video_files.sort()  # Sort for consistent ordering
                if not args.video1:
                    args.video1 = str(video_files[0])
                if not args.video2:
                    args.video2 = str(video_files[1])
                print(f"Auto-detected videos: {args.video1} and {args.video2}")
    
    # If still no videos found, prompt user
    if not args.video1 or not args.video2:
        print("\nVideo files not found. Please provide video file paths:")
        
        if not args.video1:
            while True:
                video1_path = input("Enter path to first video file: ").strip()
                if video1_path and Path(video1_path).exists():
                    args.video1 = video1_path
                    break
                print("File not found. Please try again.")
        
        if not args.video2:
            while True:
                video2_path = input("Enter path to second video file: ").strip()
                if video2_path and Path(video2_path).exists():
                    args.video2 = video2_path
                    break
                print("File not found. Please try again.")
    
    # Validate that both video files exist
    if not Path(args.video1).exists():
        print(f"Error: Video file '{args.video1}' not found!")
        sys.exit(1)
    
    if not Path(args.video2).exists():
        print(f"Error: Video file '{args.video2}' not found!")
        sys.exit(1)
    
    return args.video1, args.video2, args.output, args.fps


if __name__ == "__main__":
    print("Video Stitching Tool")
    print("=" * 50)
    
    try:
        video1, video2, output_path, fps = get_video_paths()
        
        print(f"\nProcessing videos:")
        print(f"  Video 1: {video1}")
        print(f"  Video 2: {video2}")
        print(f"  Output: {output_path}")
        print(f"  FPS: {fps}")
        print()
        
        stitch_videos(video1, video2, output_path, fps)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
