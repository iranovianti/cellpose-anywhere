"""Cellpose segmentation utilities."""

import numpy as np
from PIL import Image
from cellpose import models
import matplotlib.pyplot as plt

from .image_processing import resize_array

_cached_model = None
SEGMENTATION_SIZE = (128, 128)  # smaller size for faster processing


def get_cellpose_model(gpu=True):
    """Get or create a cached Cellpose model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = models.CellposeModel(gpu=gpu)
    return _cached_model


def run_cellpose_segmentation(image_array):
    """Run Cellpose segmentation on an image.
    
    Resizes input for faster processing, then resizes
    masks back to original dimensions.
    
    Args:
        image_array: numpy array of image (H, W) or (H, W, C)
    
    Returns:
        masks: labeled mask array (original size, int32)
    """
    orig_h, orig_w = image_array.shape[:2]
    orig_size = (orig_w, orig_h)
    
    resized = resize_array(image_array, target_size=SEGMENTATION_SIZE)
    
    model = get_cellpose_model(gpu=True)
    masks, _, _ = model.eval(resized)
    
    masks = resize_array(masks, target_size=orig_size, method='nearest').astype(np.int32)
    
    return masks


def masks_to_overlay(masks, original_image=None, alpha=0.5):
    """Convert segmentation masks to colored overlay image."""
    cmap = plt.cm.tab20
    colored = np.zeros((*masks.shape, 3), dtype=np.uint8)
    
    for obj_id in np.unique(masks):
        if obj_id == 0:
            continue
        color = (np.array(cmap(obj_id % 20)[:3]) * 255).astype(np.uint8)
        colored[masks == obj_id] = color
    
    if original_image is None:
        return Image.fromarray(colored)
    
    if original_image.ndim == 2:
        original_rgb = np.stack([original_image] * 3, axis=-1)
    else:
        original_rgb = original_image
    
    if original_rgb.dtype != np.uint8:
        orig_min, orig_max = original_rgb.min(), original_rgb.max()
        original_rgb = ((original_rgb - orig_min) / (orig_max - orig_min) * 255).astype(np.uint8)
    
    blended = (alpha * colored + (1 - alpha) * original_rgb).astype(np.uint8)
    return Image.fromarray(blended)
