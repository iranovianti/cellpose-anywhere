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
    
    # cellpose_io.save_rois creates a file with _rois.zip suffix
    output_path_base = f"{filename_base}_MASK{mask_number}"
    cellpose_io.save_rois(masks, output_path_base)
    
    # Find the actual created file
    zip_files = glob.glob(f"{output_path_base}*.zip")
    if zip_files:
        # Return absolute path
        abs_path = os.path.abspath(zip_files[0])
        return abs_path
    
    return None
