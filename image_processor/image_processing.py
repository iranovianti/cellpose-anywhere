"""Image processing utilities: normalization and display conversion."""

import numpy as np
from PIL import Image
import cv2


def normalize_to_uint8(arr):
    """Normalize a numpy array to uint8 [0, 255].

    If already uint8, returns as-is. Otherwise, scales to [0, 255].
    Handles constant arrays (min == max) by returning zeros.

    Args:
        arr: numpy array of any numeric dtype

    Returns:
        arr_uint8: normalized array as uint8
    """
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    amin, amax = a.min(), a.max()
    if amax == amin:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - amin) / (amax - amin)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a


def array_to_display_pil(arr):
    """Convert a numpy array to a PIL.Image suitable for display.

    Handles:
    - 2D arrays (grayscale) -> RGB conversion for consistent display
    - (H, W, 3) -> RGB
    - (H, W, 4) -> drop alpha, use RGB
    - (H, W, C) with C > 4 -> use first 3 channels as RGB

    Args:
        arr: numpy array of shape (H, W) or (H, W, C)

    Returns:
        pil: PIL.Image in RGB mode
    """
    if arr.ndim == 2:
        # Grayscale -> convert to RGB for consistent display
        arr8 = normalize_to_uint8(arr)
        pil = Image.fromarray(arr8, mode="L").convert("RGB")
        return pil

    if arr.ndim == 3:
        h, w, c = arr.shape
        arr8 = normalize_to_uint8(arr)
        if c == 3:
            return Image.fromarray(arr8)
        elif c == 4:
            # e.g., RGBA -> drop alpha, use RGB
            return Image.fromarray(arr8[:, :, :3])
        elif c > 4:
            # pick first 3 channels as best-effort RGB
            return Image.fromarray(arr8[:, :, :3])

    # Last-resort: try direct conversion
    return Image.fromarray(normalize_to_uint8(arr))


def resize_array(arr, scale=None, target_size=None, method='bilinear'):
    """Resize a numpy array image using OpenCV.

    Args:
        arr: numpy array of shape (H, W) or (H, W, C)
        scale: scale factor (e.g., 0.5 for half size, 2.0 for double)
        target_size: tuple of (width, height) for target dimensions
        method: interpolation method ('nearest', 'bilinear', 'bicubic', 'lanczos')

    Returns:
        Resized numpy array

    Note:
        Provide either scale or target_size, not both.
    """
    if scale is None and target_size is None:
        raise ValueError("Must provide either scale or target_size")
    if scale is not None and target_size is not None:
        raise ValueError("Provide only one of scale or target_size")

    # Map method names to cv2 constants
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
    }
    interpolation = methods.get(method, cv2.INTER_LINEAR)

    # Get original dimensions
    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w = arr.shape[:2]

    # Calculate target size
    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = target_size

    # cv2.resize handles all dtypes robustly
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
    
    return resized
