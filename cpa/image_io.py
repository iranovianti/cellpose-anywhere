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


def save_masks_as_rois(masks, filename_base, output_dir=None):
    """Save segmentation masks as ROI files.
    
    Args:
        masks: labeled mask array from Cellpose
        filename_base: base filename (without extension)
        output_dir: directory to save ROIs (defaults to temp directory)
    
    Returns:
        Path to the saved ROI zip file
    """
    from cellpose import io as cellpose_io
    import tempfile
    
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    output_path = os.path.join(output_dir, f"{filename_base}_ROI.zip")
    cellpose_io.save_rois(masks, output_path)
    
    return output_path
