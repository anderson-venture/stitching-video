#!/usr/bin/env python3
"""
Stitch two videos into a panorama (frame-by-frame).
Assumptions:
 - Videos are roughly time-synchronized (frame i corresponds to frame i).
 - There's enough overlap between frames.
 - We use one global homography computed from a frame pair (usually first pair).
"""

import sys
import cv2
import numpy as np

# ---------- Helpers: feature matching and homography ----------
def create_feature_detector():
    # Prefer SIFT if available, else ORB
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create()
    else:
        return cv2.ORB_create(5000)

def match_features(img1_gray, img2_gray, detector):
    # detect + descriptors
    kp1, des1 = detector.detectAndCompute(img1_gray, None)
    kp2, des2 = detector.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return [], [], []

    # Use FLANN for SIFT, BF for ORB
    if isinstance(detector, cv2.SIFT) or (hasattr(detector, 'descriptorType') and detector.descriptorType() == cv2.CV_32F):
        # FLANN parameters for SIFT (float descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = flann.knnMatch(des1, des2, k=2)
        # ratio test
        good = []
        for m_n in knn_matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
    else:
        # ORB - binary descriptors -> use BFMatcher with Hamming
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)
        good = sorted(matches, key=lambda x: x.distance)[:200]  # top matches

    return kp1, kp2, good

def find_homography_from_pair(img1, img2, detector):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, kp2, matches = match_features(g1, g2, detector)
    if len(matches) < 8:
        raise RuntimeError(f"Not enough matches ({len(matches)}) to compute homography")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    return H, (kp1, kp2, matches, mask)

# ---------- Helpers: exposure compensation ----------
def exposure_compensate(img_base, img_warped, H):
    """
    Simple compensation: compute the overlap region between img_base and the warped image,
    compute mean intensities and apply scale + offset to warped image to match base.
    """
    h1, w1 = img_base.shape[:2]
    h2, w2 = img_warped.shape[:2]

    # create mask for warped image non-black pixels
    mask_w = (np.sum(img_warped, axis=2) > 0).astype(np.uint8)

    # overlap mask where both base and warped have content
    _, mask_base = cv2.threshold(cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    overlap = cv2.bitwise_and(mask_w, (mask_base > 0).astype(np.uint8))

    # if overlap is tiny, skip compensation
    if cv2.countNonZero(overlap) < 50:
        return img_warped

    # compute per-channel mean in overlap
    overlap_bool = overlap.astype(bool)
    means_base = []
    means_warped = []
    for c in range(3):
        b_vals = img_base[:, :, c][overlap_bool]
        w_vals = img_warped[:, :, c][overlap_bool]
        if len(b_vals) == 0 or len(w_vals) == 0:
            means_base.append(1.0)
            means_warped.append(1.0)
        else:
            means_base.append(np.mean(b_vals))
            means_warped.append(np.mean(w_vals))

    # compute scale and offset per channel (affine: out = a*in + b -> choose b = 0 for simplicity)
    compens_img = img_warped.astype(np.float32).copy()
    for c in range(3):
        if means_warped[c] < 1e-3:
            continue
        gain = means_base[c] / (means_warped[c] + 1e-8)
        compens_img[:, :, c] = compens_img[:, :, c] * gain

    compens_img = np.clip(compens_img, 0, 255).astype(np.uint8)
    return compens_img

# ---------- Helpers: pyramid blending ----------
def pyramid_blend(img1, img2, mask, levels=5):
    """
    Multi-band blending (Laplacian pyramid). Blends img2 on top of img1 using mask.
    - mask should be float in [0,1], same size as images, 1 where img2 is kept.
    """
    # convert to float32
    G1 = img1.astype(np.float32)
    G2 = img2.astype(np.float32)
    GM = mask.astype(np.float32)

    gp1 = [G1]
    gp2 = [G2]
    gpm = [GM]
    for i in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        gpm.append(cv2.pyrDown(gpm[-1]))

    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    for i in range(levels-1, -1, -1):
        size = (gp1[i].shape[1], gp1[i].shape[0])
        GE = cv2.pyrUp(gp1[i+1], dstsize=size)
        L = cv2.subtract(gp1[i], GE)
        lp1.append(L)

        GE2 = cv2.pyrUp(gp2[i+1], dstsize=(gp2[i].shape[1], gp2[i].shape[0]))
        L2 = cv2.subtract(gp2[i], GE2)
        lp2.append(L2)

    # blend pyramids
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gpm[::-1]):
        # Handle case where mask might already have 3 channels
        if len(gm.shape) == 2:
            gm_exp = np.repeat(gm[:, :, None], 3, axis=2)
        else:
            gm_exp = gm
        ls = l1 * (1.0 - gm_exp) + l2 * gm_exp
        LS.append(ls)

    # reconstruct
    result = LS[0]
    for i in range(1, len(LS)):
        size = (LS[i].shape[1], LS[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = cv2.add(result, LS[i])

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# ---------- Stitch single pair of frames ----------
def stitch_pair(img_left, img_right, H, blend=True):
    """
    Warp right into left coordinate frame using H, then blend.
    Returns stitched image (left anchored at top-left).
    """
    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]

    # get corners of right image after transform
    corners_right = np.float32([[0,0], [wR,0], [wR,hR], [0,hR]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_right, H)
    all_corners = np.vstack((np.float32([[0,0],[wL,0],[wL,hL],[0,hL]]).reshape(-1,1,2), warped_corners))

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    offset_x = -xmin
    offset_y = -ymin

    # translation matrix to place images in positive canvas coords
    trans = np.array([[1,0,offset_x], [0,1,offset_y], [0,0,1]])

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    # warp right image
    warped_right = cv2.warpPerspective(img_right, trans.dot(H), (canvas_w, canvas_h))
    canvas_left = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_left[offset_y:offset_y+hL, offset_x:offset_x+wL] = img_left

    if not blend:
        # simple overlay: put left then for non-zero pixels of warped_right replace
        mask_r = (np.sum(warped_right, axis=2) > 0)
        canvas_left[mask_r] = warped_right[mask_r]
        return canvas_left

    # exposure compensation
    warped_right = exposure_compensate(canvas_left, warped_right, trans.dot(H))

    # build mask for warped_right (1 where valid)
    mask_r = (np.sum(warped_right, axis=2) > 0).astype(np.uint8)
    mask_l = (np.sum(canvas_left, axis=2) > 0).astype(np.uint8)
    # create blend mask so that in overlap we smoothly mix
    overlap = (mask_l & mask_r).astype(np.uint8)
    # start with mask where warped_right dominates on its side of seam
    # compute distance transform to generate soft mask
    dist_right = cv2.distanceTransform((mask_r*255).astype(np.uint8), cv2.DIST_L2, 5)
    dist_left = cv2.distanceTransform((mask_l*255).astype(np.uint8), cv2.DIST_L2, 5)
    # avoid division by zero
    denom = (dist_right + dist_left).astype(np.float32)
    denom[denom == 0] = 1.0
    soft_mask = (dist_right / denom)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    mask_float = np.repeat(soft_mask[:, :, None], 3, axis=2).astype(np.float32)
    blended = pyramid_blend(canvas_left, warped_right, mask_float, levels=5)
    return blended

# ---------- Process videos ----------
def stitch_videos(left_path, right_path, out_path, out_fps=None, use_blend=True, use_global_homography=True):
    capL = cv2.VideoCapture(left_path)
    capR = cv2.VideoCapture(right_path)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Cannot open one of the input videos")

    fpsL = capL.get(cv2.CAP_PROP_FPS) or 30
    fpsR = capR.get(cv2.CAP_PROP_FPS) or 30
    # Since user said time sync is perfect, assume we can iterate frame-by-frame.
    fps = out_fps if out_fps is not None else min(fpsL, fpsR)

    # compute frame counts and use min length
    framesL = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))
    framesR = int(capR.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(framesL, framesR)

    # read first valid pair to compute homography
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        raise RuntimeError("Failed to read first frames")

    detector = create_feature_detector()
    print("Computing global homography (using first frame pair)...")
    try:
        H, _ = find_homography_from_pair(frameL, frameR, detector)
    except Exception as e:
        print("First-pair homography failed:", e)
        # try a few frames ahead
        success = False
        capL.set(cv2.CAP_PROP_POS_FRAMES, 0)
        capR.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for trial in range(1, min(20, total)):
            capL.set(cv2.CAP_PROP_POS_FRAMES, trial)
            capR.set(cv2.CAP_PROP_POS_FRAMES, trial)
            retL, fL = capL.read()
            retR, fR = capR.read()
            if not retL or not retR:
                break
            try:
                H, _ = find_homography_from_pair(fL, fR, detector)
                success = True
                frameL = fL
                frameR = fR
                print(f"Found homography using frame {trial}")
                break
            except Exception:
                continue
        if not success:
            raise RuntimeError("Failed to compute homography from initial frames. Check overlap.")

    # prepare writer by stitching first pair to get size
    stitched0 = stitch_pair(frameL, frameR, H, blend=use_blend)
    hS, wS = stitched0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (wS, hS))
    if not writer.isOpened():
        raise RuntimeError("Cannot open output video writer")

    # write first frame
    writer.write(stitched0)
    print(f"Output video: {out_path}, size: {wS}x{hS}, fps: {fps}")

    # reset to frame 1 (we wrote frame 0 already)
    capL.set(cv2.CAP_PROP_POS_FRAMES, 1)
    capR.set(cv2.CAP_PROP_POS_FRAMES, 1)
    frame_index = 1
    try:
        while frame_index < total:
            retL, fL = capL.read()
            retR, fR = capR.read()
            if (not retL) or (not retR):
                break

            # stitch using same H (robust and faster). If you expect large camera motion, you can recompute H occasionally.
            stitched = stitch_pair(fL, fR, H, blend=use_blend)
            # If stitched size differs from first (rare), resize or pad
            if stitched.shape[:2] != (hS, wS):
                stitched = cv2.resize(stitched, (wS, hS))

            writer.write(stitched)
            if frame_index % 50 == 0:
                print(f"Processed frame {frame_index}/{total}")
            frame_index += 1
    finally:
        capL.release()
        capR.release()
        writer.release()
    print("Stitching complete.")

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python stitch_videos_panorama.py left_video.mp4 right_video.mp4 output.mp4 [out_fps]")
        sys.exit(1)
    left_vid = sys.argv[1]
    right_vid = sys.argv[2]
    out_vid = sys.argv[3]
    out_fps = None
    if len(sys.argv) >= 5:
        out_fps = float(sys.argv[4])

    stitch_videos(left_vid, right_vid, out_vid, out_fps=out_fps, use_blend=True)
