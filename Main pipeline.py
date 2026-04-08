# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Spyder Editor

This is a temporary script file.
"""
# This future import changes how type annotations are handled.
# The triple-quoted text below is just a docstring / note.
# It does not affect how the script runs.
"""
Week 3 - IDE Script (P0028 bent-head project)

This script:
1) Recursively reads PNG images from the Week 1 folder
   (baseline / 48hrs / 96hrs)
2) Determines timepoint from folder name (fallback: filename)
3) Extracts oil, cauda, and replicate from filename
4) Builds a CSV summary (one row per image)
5) Saves example outputs from ONE image:
   - grayscale
   - Gaussian blur
   - overlay of detected sperm heads

Verbose + commented for teaching/debugging.
"""
# This block is the script description.
# It explains the overall goals of the pipeline:
# find images, extract metadata, and save example image outputs.

import re

from pathlib import Path
from typing import List, Dict, Any, Optional

# re is used for regular expression matching in filenames.
# Path is used to work with folders and file paths.
# The typing imports are just for type hints and readability.

import numpy as np
import pandas as pd

# numpy is used for image/math operations.
# pandas is used to build the summary table and save it as a CSV.

import sys
print(sys.executable)
import skimage
print(skimage.__version__)

from skimage import io, exposure
from skimage.filters import gaussian
from skimage.feature import blob_log
from skimage.util import img_as_float
from skimage.color import rgb2gray, rgba2rgb

# Toolkit imports
from bent_utils import (
    ensure_week4_dirs,
    save_mask_png,
    crop_roi,
    save_roi_png,
    segment_sperm_in_roi,
    skeletonize_mask,
    save_skeleton_png,
    compute_bend_angle_deg,
    predict_bent,
    save_roi_overlay_png,
    ROI_SIZE,
    BEND_CUTOFF_DEG,
)
# ----------------------------
# skimage provides image processing tools.
# io reads and writes images, gaussian smooths images,
# blob_log detects sperm-head-like blobs,
# and rgb2gray/img_as_float convert images into analysis-ready format.
# ----------------------------

# Root folder containing: baseline/, 48hrs/, 96hrs/ (or similar)
from pathlib import Path
DATA_ROOT = Path("/Users/kendalllaberge/Downloads/Kendall_CV_pipeline_project")

# DATA_ROOT points to the main folder that contains the raw image folders.
# This is the top-level folder the script will search through for PNG images.

# Output folder (kept separate from raw data)
OUTPUT_ROOT = DATA_ROOT / "outputs_spyder_week4"

# OUTPUT_ROOT is where all generated results will be saved.
# This keeps output files separate from the raw image data.

IMAGE_EXT = ".png"
CSV_NAME = "week1_summary.csv"

# IMAGE_EXT tells the script which image type to search for.
# CSV_NAME sets the filename for the metadata summary output.

# Which image is used to generate example outputs
EXAMPLE_IMAGE_INDEX = 0

# This selects which image in the dataset will be used
# to create the example grayscale, blur, and overlay outputs.

# Image processing parameters
BG_SIGMA = 25.0            # Background estimation blur
EXAMPLE_BLUR_SIGMA = 2.0   # Small blur for the example output

# BG_SIGMA controls the large blur used for background subtraction.
# EXAMPLE_BLUR_SIGMA controls the lighter blur used only for the example output image.

# Blob detection tuned for ~7 px diameter heads
BLOB_MIN_SIGMA = 1.8
BLOB_MAX_SIGMA = 3.2
BLOB_NUM_SIGMA = 10

# These parameters control the blob detector.
# They define the size range of blobs to search for
# and how many scales are tested between the minimum and maximum sigma values.

# Increase this if overlay shows too many detections (main tuning knob)
BLOB_THRESHOLD = 0.20

# Safety cap so overlays don't turn solid red
MAX_BLOBS_TO_DRAW = 300


# ----------------------------
# PARSING HELPERS
# ----------------------------

def has_token(text: str, token: str) -> bool:
    """
    Robustly detect tokens like MO, PO, C1, C2 even when surrounded
    by underscores, hyphens, or other non-alphanumeric characters.
    """
    pattern = rf"(^|[^A-Za-z0-9]){re.escape(token)}([^A-Za-z0-9]|$)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

# This helper function checks whether a specific token appears in a filename.
# It is designed to detect standalone labels like C1, C2, MO, or PO
# without accidentally matching them inside longer strings.

def parse_timepoint_from_folder(folder_name: str, filename: str) -> str:
    """
    Determine timepoint.
    Priority:
      1) folder name
      2) filename fallback
    """
    fld = folder_name.lower()
    fn = filename.lower()

    # Folder-based
    if "baseline" in fld or fld.startswith("base"):
        return "baseline"
    if "48" in fld:
        return "48"
    if "96" in fld:
        return "96"

    # Filename fallback
    if "baseline" in fn or fn.startswith("base"):
        return "baseline"
    if "48hrs" in fn or re.search(r"(?<!\d)48(?!\d)", fn):
        return "48"
    if "96hrs" in fn or re.search(r"(?<!\d)96(?!\d)", fn):
        return "96"

    return "unknown"


def timepoint_to_hours(tp: str) -> Optional[int]:
    """Convert timepoint label to numeric hours."""
    if tp == "baseline":
        return 0
    if tp == "48":
        return 48
    if tp == "96":
        return 96
    return None


def parse_group_from_filename(filename: str) -> str:
    """
    Parse group token from filename.
    Returns: C1, C2, MO, PO, or unknown.
    """
    if has_token(filename, "C1"):
        return "C1"
    if has_token(filename, "C2"):
        return "C2"
    if has_token(filename, "MO"):
        return "MO"
    if has_token(filename, "PO"):
        return "PO"
    return "unknown"


def link_group_to_oil(group: str) -> str:
    """Map baseline groups to oil pairing."""
    g = group.upper()
    if g in ("C1", "MO"):
        return "MO"
    if g in ("C2", "PO"):
        return "PO"
    return "unknown"


def infer_cauda(group: str, oil: str) -> str:
    """
    Cauda rules:
    - If filename explicitly says C1 or C2 → trust it
    - Otherwise infer from oil (MO → C1, PO → C2)
    """
    g = group.upper()
    o = oil.upper()
    if g in ("C1", "C2"):
        return g
    if o == "MO":
        return "C1"
    if o == "PO":
        return "C2"
    return "unknown"


def parse_replicate(filename: str) -> str:
    """Extract replicate label R1/R2/R3 if present."""
    m = re.search(r"(^|[^A-Za-z0-9])R(\d+)([^A-Za-z0-9]|$)", filename, flags=re.IGNORECASE)
    if m:
        return f"R{m.group(2)}"
    return "unknown"


# ----------------------------
# IO + IMAGE OPS
# ----------------------------

def ensure_output_dirs(root: Path) -> Dict[str, Path]:
    """Create output directories and return output paths."""
    root.mkdir(parents=True, exist_ok=True)
    ex = root / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    return {
        "csv": root / CSV_NAME,
        "gray": ex / "example_grayscale.png",
        "gauss": ex / "example_gaussian_blur.png",
        "overlay": ex / "example_overlay_detection.png",
    }


def find_images(root: Path, ext: str) -> List[Path]:
    """Find all images recursively, excluding generated output folders."""
    paths = []
    for p in root.rglob(f"*{ext}"):
        if OUTPUT_ROOT in p.parents:
            continue
        paths.append(p)
    return sorted(paths)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float [0,1]."""
    if img.ndim == 2:
        return img_as_float(img)

    if img.ndim == 3:
        # Standard RGB image
        if img.shape[2] == 3:
            return rgb2gray(img_as_float(img))

        # RGBA image (RGB + alpha channel)
        if img.shape[2] == 4:
            return rgb2gray(rgba2rgb(img_as_float(img)))

    raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")


def preprocess_negative_phase(gray: np.ndarray) -> np.ndarray:
    """
    Background subtraction + contrast rescaling
    for negative phase images.
    """
    bg = gaussian(gray, sigma=BG_SIGMA, preserve_range=True)
    corr = gray - bg
    p2, p98 = np.percentile(corr, (2, 98))
    return exposure.rescale_intensity(corr, in_range=(p2, p98), out_range=(0, 1))


def detect_heads(img_proc: np.ndarray) -> np.ndarray:
    """Detect sperm heads using LoG blob detection."""
    return blob_log(
        img_proc,
        min_sigma=BLOB_MIN_SIGMA,
        max_sigma=BLOB_MAX_SIGMA,
        num_sigma=BLOB_NUM_SIGMA,
        threshold=BLOB_THRESHOLD,
    )


def save_image(path: Path, img01: np.ndarray) -> None:
    """Save float image [0,1] as PNG."""
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    io.imsave(str(path), img8)


def save_overlay(path: Path, img_proc: np.ndarray, blobs: np.ndarray) -> None:
    """
    Draw red rings for detected blobs.
    Only draw up to MAX_BLOBS_TO_DRAW.
    """
    base = (np.clip(img_proc, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)

    blobs = blobs[:MAX_BLOBS_TO_DRAW]
    h, w = base.shape

    for y, x, sigma in blobs:
        r = int(round(sigma * np.sqrt(2)))
        cy, cx = int(round(y)), int(round(x))

        # ring pixels
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d2 = dy * dy + dx * dx
                if (r - 1) ** 2 <= d2 <= (r + 1) ** 2:
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        rgb[yy, xx] = [255, 0, 0]

    io.imsave(str(path), rgb)


# ----------------------------
# MAIN PIPELINE
# ----------------------------

# The main pipeline coordinates the full workflow.
# It finds all images, extracts metadata, saves a summary CSV,
# processes an example image, detects sperm heads using blob detection,
# and generates example output images for debugging.

def main() -> None:
    out = ensure_output_dirs(OUTPUT_ROOT)

    if not DATA_ROOT.exists():
        raise RuntimeError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    image_paths = find_images(DATA_ROOT, IMAGE_EXT)
    if not image_paths:
        raise RuntimeError(f"No {IMAGE_EXT} images found under: {DATA_ROOT}")

    print(f"Found {len(image_paths)} images")

    rows: List[Dict[str, Any]] = []
    for p in image_paths:
        tp_label = parse_timepoint_from_folder(p.parent.name, p.name)
        tp_hr = timepoint_to_hours(tp_label)

        group = parse_group_from_filename(p.name)
        oil = link_group_to_oil(group)
        cauda = infer_cauda(group, oil)
        rep = parse_replicate(p.name)

        rows.append(
            {
                "filename": p.name,
                "filepath": str(p),
                "folder": p.parent.name,
                "timepoint": tp_label,
                "timepoint_hr": tp_hr,
                "group": group,
                "oil": oil,
                "cauda": cauda,
                "replicate": rep,
                "qc_missing_timepoint": tp_hr is None,
                "qc_missing_group": group == "unknown",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out["csv"], index=False)
    print(f"Saved CSV → {out['csv']}")

    # Example outputs
    ex_idx = min(EXAMPLE_IMAGE_INDEX, len(image_paths) - 1)
    ex_path = image_paths[ex_idx]
    print(f"Creating example outputs from: {ex_path}")

    raw = io.imread(str(ex_path))
    gray = to_grayscale(raw)
    save_image(out["gray"], gray)

    gauss = gaussian(gray, sigma=EXAMPLE_BLUR_SIGMA, preserve_range=True)
    save_image(out["gauss"], gauss)

    proc = preprocess_negative_phase(gray)
    blobs = detect_heads(proc)
    print(f"Detected {len(blobs)} blobs (drawing {min(len(blobs), MAX_BLOBS_TO_DRAW)})")

     # ------------------------------------------------------------
    # WEEK 4: ROI -> MASK -> SKELETON -> BEND ANGLE -> CLASSIFY
    # ------------------------------------------------------------
    week4 = ensure_week4_dirs(OUTPUT_ROOT)

    bend_rows: List[Dict[str, Any]] = []

    for p in image_paths:
        raw = io.imread(str(p))
        gray = to_grayscale(raw)
        proc = preprocess_negative_phase(gray)

        blobs = detect_heads(proc)

        for i, (y, x, sigma) in enumerate(blobs):
            cy, cx = int(round(y)), int(round(x))

            roi, qc_oob = crop_roi(proc, cy, cx, ROI_SIZE)
            if qc_oob or roi is None:
                bend_rows.append(
                    {
                        "image": p.name,
                        "filepath": str(p),
                        "head_idx": i,
                        "cy": cy,
                        "cx": cx,
                        "sigma": float(sigma),
                        "qc_oob": True,
                        "mask_area": None,
                        "thr": None,
                        "inverted": None,
                        "bend_angle_deg": np.nan,
                        "bent": None,
                        "reason": "roi_oob",
                    }
                )
                continue

            stem = f"{p.stem}_h{i:03d}"
            roi_path = week4["rois"] / f"{stem}.png"
            mask_path = week4["masks"] / f"{stem}_mask.png"
            skel_path = week4["skels"] / f"{stem}_skel.png"
            overlay_path = week4["overlays"] / f"{stem}_overlay.png"

            save_roi_png(roi_path, roi)

            mask, qc_mask = segment_sperm_in_roi(roi)
            save_mask_png(mask_path, mask)

            skel = skeletonize_mask(mask)
            save_skeleton_png(skel_path, skel)

            bend_angle, qc_bend = compute_bend_angle_deg(roi, mask, skel)
            bent = predict_bent(bend_angle, BEND_CUTOFF_DEG)

            save_roi_overlay_png(
                overlay_path,
                roi,
                skel,
                head_center_xy=(ROI_SIZE // 2, ROI_SIZE // 2),
            )

            bend_rows.append(
                {
                    "image": p.name,
                    "filepath": str(p),
                    "head_idx": i,
                    "cy": cy,
                    "cx": cx,
                    "sigma": float(sigma),
                    "qc_oob": False,
                    "mask_area": qc_mask.get("mask_area"),
                    "thr": qc_mask.get("thr"),
                    "inverted": qc_mask.get("inverted"),
                    "bend_angle_deg": bend_angle,
                    "bent": bent,
                    "reason": qc_bend.get("reason", ""),
                }
            )

        print(f"{p.name}: {len(blobs)} heads processed")

    bend_df = pd.DataFrame(bend_rows)
    bend_df.to_csv(week4["csv"], index=False)
    print(f"Saved bend features CSV → {week4['csv']}")
    print("All outputs saved successfully.")


if __name__ == "__main__":
    main()
