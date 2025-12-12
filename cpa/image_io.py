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


def save_masks_as_rois(masks, filename_base, mask_number=1):
    """Save segmentation masks as ROI files for download.
    
    Args:
        masks: labeled mask array from Cellpose
        filename_base: base filename (without extension)
        mask_number: mask index (1-based) for suffix naming
    
    Returns:
        Absolute path to the saved ROI zip file
    """
    from cellpose import io as cellpose_io
    import glob
    
    # Sanitize filename: replace spaces and special characters
    safe_filename = filename_base.replace(" ", "_")
    
    # cellpose_io.save_rois creates a file with _rois.zip suffix
    output_path_base = f"{safe_filename}_MASK{mask_number}"
    cellpose_io.save_rois(masks, output_path_base)
    
    # Find the actual created file (escape glob special chars)
    escaped_path = glob.escape(output_path_base)
    zip_files = glob.glob(f"{escaped_path}*.zip")
    if zip_files:
        # Return absolute path
        abs_path = os.path.abspath(zip_files[0])
        return abs_path
    
    return None
