#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:36:06 2026

@author: kendalllaberge
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from skimage import io
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, skeletonize, binary_opening, disk
from skimage.measure import label, regionprops


ROI_SIZE = 128
BEND_CUTOFF_DEG = 35.0
WEEK4_ROOTNAME = "week4_outputs"


# ----------------------------
# OUTPUT FOLDERS
# ----------------------------

def ensure_week4_dirs(output_root: Path) -> Dict[str, Path]:
    """
    Create a new output folder inside output_root / WEEK4_ROOTNAME and subfolders.
    """
    root = output_root / WEEK4_ROOTNAME
    rois = root / "rois"
    masks = root / "masks"
    skels = root / "skeletons"
    overlays = root / "overlays"
    csv_dir = root / "csv"

    for p in [root, rois, masks, skels, overlays, csv_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "rois": rois,
        "masks": masks,
        "skels": skels,
        "overlays": overlays,
        "csv": csv_dir / "bend_features_week4.csv",
    }


# ----------------------------
# ROI CROPPING
# ----------------------------

def crop_roi(gray_full: np.ndarray, cy: int, cx: int, roi_size: int) -> Tuple[Optional[np.ndarray], bool]:
    """
    Crop roi_size x roi_size ROI centered at (cy, cx). If out-of-bounds, return (None, True).
    """
    h, w = gray_full.shape[:2]
    half = roi_size // 2

    y0 = cy - half
    y1 = cy + half
    x0 = cx - half
    x1 = cx + half

    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None, True

    roi = gray_full[y0:y1, x0:x1]
    if roi.shape != (roi_size, roi_size):
        return None, True

    return roi, False


def save_roi_png(roi_path: Path, roi_gray01: np.ndarray) -> None:
    """
    Save ROI grayscale float image [0,1] as 8-bit PNG.
    """
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    img8 = (np.clip(roi_gray01, 0, 1) * 255).astype(np.uint8)
    io.imsave(str(roi_path), img8)


# ----------------------------
# ROI SEGMENTATION (MASK)
# ----------------------------

def segment_sperm_in_roi(roi_gray01: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a binary sperm mask in the ROI using smoothing + Otsu + cleanup.
    For your current images, the sperm appears bright, so we use the bright mask.
    Then we keep the center-nearest elongated component.
    """
    qc: Dict[str, Any] = {}

    # 1) smooth
    sm = gaussian(roi_gray01, sigma=1.5, preserve_range=True)

    # 2) threshold (Otsu)
    thr = threshold_otsu(sm)
    qc["thr"] = float(thr)

    # Try both polarities for QC, but use bright mask for these images
    mask_dark = sm < thr
    mask_bright = sm > thr

    # 3) cleanup helper
    def clean(m: np.ndarray) -> np.ndarray:
        m2 = remove_small_objects(m, min_size=10)
        m2 = remove_small_holes(m2, area_threshold=10)
        return m2.astype(bool)

    md = clean(mask_dark)
    mb = clean(mask_bright)

    qc["area_dark"] = int(md.sum())
    qc["area_bright"] = int(mb.sum())

    # Use bright foreground
    qc["inverted"] = False
    mask = mb

        # Clean binary mask a bit
    mask = remove_small_objects(mask, min_size=20)
    mask = remove_small_holes(mask, area_threshold=20)

    # Break weak noisy bridges before connected-component selection
    mask = binary_opening(mask, disk(2))

    # -------------------------------------------------
    # KEEP ONLY THE COMPONENT CONNECTED TO THE CENTER
    # -------------------------------------------------
    lbl = label(mask)

    h, w = mask.shape
    cy0 = h // 2
    cx0 = w // 2

    # If center pixel is already foreground, use it directly
    if mask[cy0, cx0]:
        center_label = lbl[cy0, cx0]
    else:
        # Otherwise find the foreground pixel closest to the center
        ys, xs = np.nonzero(mask)

        if len(ys) == 0:
            qc["mask_area"] = 0
            return mask, qc

        d2 = (ys - cy0) ** 2 + (xs - cx0) ** 2
        k = np.argmin(d2)
        nearest_y = ys[k]
        nearest_x = xs[k]
        center_label = lbl[nearest_y, nearest_x]

    # Keep only that connected component
    mask = (lbl == center_label)

    qc["mask_area"] = int(mask.sum())
    return mask, qc

def save_mask_png(mask_path: Path, mask: np.ndarray) -> None:
    """
    Save binary mask as 0/255 PNG.
    """
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask8 = (mask.astype(np.uint8) * 255)
    io.imsave(str(mask_path), mask8)


# ----------------------------
# SKELETONIZATION
# ----------------------------
def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary mask to a 1-pixel-wide centerline,
    then keep only the main path from the center-near point
    to the farthest endpoint.
    """
    skel = skeletonize(mask).astype(bool)
    skel = keep_main_skeleton_path(skel)
    return skel

def keep_main_skeleton_path(skel: np.ndarray) -> np.ndarray:
    """
    Keep only the main skeleton path from the skeleton point nearest the ROI center
    to the farthest endpoint. This removes tail-end side branches.
    """
    import numpy as np

    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return skel

    h, w = skel.shape
    cy0 = h // 2
    cx0 = w // 2

    coords = list(zip(ys, xs))
    coord_set = set(coords)

    # 8-neighborhood graph
    neighbors = {}
    for y, x in coords:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if (yy, xx) in coord_set:
                    nbrs.append((yy, xx))
        neighbors[(y, x)] = nbrs

    # endpoints = skeleton pixels with exactly 1 neighbor
    endpoints = [p for p, nbrs in neighbors.items() if len(nbrs) == 1]
    if len(endpoints) == 0:
        return skel

    # start at skeleton pixel nearest ROI center
    d2_center = [(y - cy0) ** 2 + (x - cx0) ** 2 for y, x in coords]
    start = coords[int(np.argmin(d2_center))]

    # BFS shortest paths from start
    from collections import deque

    parent = {start: None}
    dist = {start: 0}
    q = deque([start])

    while q:
        cur = q.popleft()
        for nb in neighbors[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                q.append(nb)

    reachable_endpoints = [p for p in endpoints if p in dist]
    if len(reachable_endpoints) == 0:
        return skel

    # choose farthest endpoint from start
    end = max(reachable_endpoints, key=lambda p: dist[p])

    # trace path back
    keep = np.zeros_like(skel, dtype=bool)
    cur = end
    while cur is not None:
        keep[cur] = True
        cur = parent[cur]

    return keep

def save_skeleton_png(skel_path: Path, skel: np.ndarray) -> None:
    """
    Save skeleton as 0/255 PNG.
    """
    skel_path.parent.mkdir(parents=True, exist_ok=True)
    skel8 = (skel.astype(np.uint8) * 255)
    io.imsave(str(skel_path), skel8)


# ----------------------------
# BEND FEATURE + CLASSIFICATION
# ----------------------------

def _angle_from_vector(vy: float, vx: float) -> float:
    """
    Return an angle in degrees in [0, 180) for a direction vector.
    """
    ang = math.degrees(math.atan2(vy, vx))  # (-180, 180]
    if ang < 0:
        ang += 180.0
    return ang


def _smallest_angle_deg(a: float, b: float) -> float:
    """
    Smallest difference between angles in [0,180): returns [0,90]
    """
    d = abs(a - b)
    d = min(d, 180.0 - d)
    return min(d, 180.0 - d)


def compute_bend_angle_deg(
    roi_gray01: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute bend angle by comparing:
    - head major-axis orientation (regionprops) vs
    - neck direction from nearby skeleton points (PCA line fit).
    """
    qc: Dict[str, Any] = {}

    if mask is None or mask.sum() == 0:
        return float("nan"), {"reason": "empty_mask"}

    if skel is None or skel.sum() == 0:
        return float("nan"), {"reason": "empty_skeleton"}

    # A) Head axis angle using regionprops on largest component near center
    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return float("nan"), {"reason": "no_regions"}

    cy0 = roi_gray01.shape[0] // 2
    cx0 = roi_gray01.shape[1] // 2

    # pick region whose centroid is closest to ROI center, tie-break by area
    props_sorted = sorted(
        props,
        key=lambda r: ( (r.centroid[0]-cy0)**2 + (r.centroid[1]-cx0)**2, -r.area )
    )
    head = props_sorted[0]

    # skimage orientation: angle between x-axis and major axis (radians), range (-pi/2, pi/2)
    # We'll convert to degrees in [0,180)
    head_ang = math.degrees(head.orientation)
    # orientation sign convention differs; make it comparable as undirected axis
    if head_ang < 0:
        head_ang += 180.0
    qc["head_angle_deg"] = float(head_ang)
    qc["head_area"] = int(head.area)
    qc["head_centroid"] = (float(head.centroid[0]), float(head.centroid[1]))

    # B) Neck direction: skeleton points near head/center, fit line via PCA
    ys, xs = np.nonzero(skel)
    pts = np.column_stack([ys, xs]).astype(float)
    qc["skel_points"] = int(len(pts))

    if len(pts) < 12:
        return float("nan"), {"reason": "few_skeleton_points", "skel_points": int(len(pts))}

    # choose points near head centroid (within radius R)
    hy, hx = head.centroid
    d2 = (pts[:, 0] - hy) ** 2 + (pts[:, 1] - hx) ** 2
    R = 25.0  # pixels, adjust later
    near = pts[d2 <= R * R]

    qc["near_points"] = int(len(near))
    if len(near) < 8:
        return float("nan"), {"reason": "few_near_skeleton_points", "near_points": int(len(near))}

    # PCA: direction of maximum variance
    mean = near.mean(axis=0)
    X = near - mean
    cov = (X.T @ X) / max(len(near) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # [vy, vx] in (y,x) coords

    neck_ang = _angle_from_vector(vy=float(v[0]), vx=float(v[1]))
    qc["neck_angle_deg"] = float(neck_ang)

    # C) Bend angle (smallest angle between undirected axes)
    bend = _smallest_angle_deg(head_ang, neck_ang)
    qc["bend_angle_deg"] = float(bend)

    return float(bend), qc


def predict_bent(bend_angle_deg: float, cutoff: float) -> Optional[int]:
    """
    Return 1 if bent, 0 if not bent, None if NaN.
    """
    if bend_angle_deg is None or np.isnan(bend_angle_deg):
        return None
    return 1 if bend_angle_deg > cutoff else 0


# ----------------------------
# OVERLAY DEBUG IMAGE
# ----------------------------

def save_roi_overlay_png(
    overlay_path: Path,
    roi_gray01: np.ndarray,
    skel: np.ndarray,
    head_center_xy: Tuple[int, int],
) -> None:
    """
    Create overlay: grayscale ROI as RGB + skeleton in red + head center in green.
    """
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    base = (np.clip(roi_gray01, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)

    # skeleton in red
    if skel is not None:
        yy, xx = np.nonzero(skel)
        rgb[yy, xx] = [255, 0, 0]

    # head center mark in green
    cx, cy = head_center_xy  # (x,y)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y = cy + dy
            x = cx + dx
            if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[y, x] = [0, 255, 0]

    io.imsave(str(overlay_path), rgb)