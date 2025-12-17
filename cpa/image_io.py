"""Image I/O helpers for reading various image formats."""

import numpy as np
from PIL import Image
import os

# Try to import tifffile for robust TIFF support; fall back gracefully
try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None
    _HAS_TIFFFILE = False


def read_image_array(file):
    """Read a file into a numpy array.

    Supports TIFF via tifffile when available, and other formats via Pillow.
    Auto-detects and transposes (C, H, W) format to (H, W, C) when C ≤ 5.

    Args:
        file: File object with a `name` attribute (e.g., from gr.File)

    Returns:
        arr: numpy array of shape (H, W) or (H, W, C)

    Raises:
        ValueError: If file has no name attribute
    """
    name = getattr(file, "name", None)
    if name is None:
        raise ValueError("File object has no name attribute")
    ext = os.path.splitext(name)[1].lower()

    if ext in (".tif", ".tiff") and _HAS_TIFFFILE:
        arr = tifffile.imread(name)
    else:
        pil = Image.open(name)
        arr = np.array(pil)

    arr = np.squeeze(arr)
    
    # Auto-detect and transpose (C, H, W) → (H, W, C)
    # Assume channels-first if first dimension is ≤ 5 and smaller than spatial dims
    if arr.ndim == 3 and arr.shape[0] <= 5 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))
    
    return arr


def read_image_stack(file):
    """Read a file into a list of numpy arrays (one per frame/slice).

    Supports single images and stacks (timelapse, Z-stack).
    Auto-detects N dimension and returns a list of (H, W) or (H, W, C) arrays.

    Args:
        file: File object with a `name` attribute (e.g., from gr.File)

    Returns:
        frames: list of numpy arrays, each of shape (H, W) or (H, W, C)
        
    Dimension detection:
        (H, W)         → single grayscale         → [arr]
        (H, W, C)      → single multi-channel     → [arr]  (C ≤ 5)
        (C, H, W)      → single multi-channel     → [arr]  (C ≤ 5, transposed)
        (N, H, W)      → N grayscale frames       → [frame0, frame1, ...]  (N > 5)
        (N, C, H, W)   → N multi-channel frames   → [frame0, frame1, ...]  (C ≤ 5)
        (N, H, W, C)   → N multi-channel frames   → [frame0, frame1, ...]  (C ≤ 5)

    Raises:
        ValueError: If file has no name attribute or unsupported dimensions
    """
    name = getattr(file, "name", None)
    if name is None:
        raise ValueError("File object has no name attribute")
    ext = os.path.splitext(name)[1].lower()

    if ext in (".tif", ".tiff") and _HAS_TIFFFILE:
        arr = tifffile.imread(name)
    else:
        pil = Image.open(name)
        arr = np.array(pil)

    arr = np.squeeze(arr)
    ndim = arr.ndim

    # 2D: single grayscale image
    if ndim == 2:
        return [arr]

    # 3D: could be (H, W, C), (C, H, W), or (N, H, W)
    if ndim == 3:
        d0, d1, d2 = arr.shape
        
        # (H, W, C) - channels-last, single image
        if d2 <= 5 and d2 < min(d0, d1):
            return [arr]
        
        # (C, H, W) - channels-first, single image → transpose
        if d0 <= 5 and d0 < min(d1, d2):
            return [np.transpose(arr, (1, 2, 0))]
        
        # (N, H, W) - N grayscale frames
        # N > 5 distinguishes from channels
        return [arr[i] for i in range(d0)]

    # 4D: (N, C, H, W) or (N, H, W, C)
    if ndim == 4:
        d0, d1, d2, d3 = arr.shape
        
        # (N, H, W, C) - channels-last
        if d3 <= 5 and d3 < min(d1, d2):
            return [arr[i] for i in range(d0)]
        
        # (N, C, H, W) - channels-first → transpose each frame
        if d1 <= 5 and d1 < min(d2, d3):
            return [np.transpose(arr[i], (1, 2, 0)) for i in range(d0)]
        
        raise ValueError(f"Cannot interpret 4D array shape {arr.shape}")

    raise ValueError(f"Unsupported array dimensions: {ndim}D")


def save_masks_as_rois(masks, filename_base, mask_number=1):
    """Save segmentation masks as ROI files for download.
    
    Handles both single masks and lists of masks (for stacks).
    For stacks, creates a single zip with ROIs that have position set per frame.
    
    Args:
        masks: labeled mask array from Cellpose, or list of mask arrays for stacks
        filename_base: base filename (without extension)
        mask_number: mask index (1-based) for suffix naming
    
    Returns:
        Absolute path to the saved ROI zip file
    """
    from cellpose import utils
    from roifile import ImagejRoi, roiwrite
    import re
    
    # Sanitize filename: keep only alphanumeric, underscores, and hyphens
    safe_filename = re.sub(r'[^\w\-]', '_', filename_base)
    output_zip = f"{safe_filename}_MASK{mask_number}_rois.zip"
    
    # Normalize to list
    if not isinstance(masks, list):
        masks = [masks]
    
    all_rois = []
    
    for frame_idx, frame_mask in enumerate(masks):
        if frame_mask.max() == 0:
            continue
            
        outlines = utils.outlines_list(frame_mask)
        n_frames = len(masks)
        
        for label, outline in zip(np.unique(frame_mask)[1:], outlines):
            if len(outline) > 0:
                # Include frame number in name if multi-frame
                if n_frames > 1:
                    name = f"f{frame_idx+1:03d}_c{label:04d}"
                    roi = ImagejRoi.frompoints(outline, name=name, position=frame_idx+1)
                else:
                    name = f"c{label:04d}"
                    roi = ImagejRoi.frompoints(outline, name=name)
                all_rois.append(roi)
    
    if all_rois:
        roiwrite(output_zip, all_rois, mode='w')
        return os.path.abspath(output_zip)
    
    return None
