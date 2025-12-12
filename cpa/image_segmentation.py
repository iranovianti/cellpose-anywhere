"""Cellpose segmentation utilities."""

import numpy as np
from PIL import Image
from cellpose import models, utils
import matplotlib.pyplot as plt
import cv2

from .image_processing import resize_array

_cached_model = None


def get_cellpose_model(gpu=True):
    """Get or create a cached Cellpose model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = models.CellposeModel(gpu=gpu)
    return _cached_model


def run_cellpose_segmentation(image_array, segmentation_size=512):
    """Run Cellpose segmentation on an image.
    
    Resizes input for faster processing, then resizes
    masks back to original dimensions.
    
    Args:
        image_array: numpy array of image (H, W) or (H, W, C)
        segmentation_size: size to resize image to for segmentation (default 512)
    
    Returns:
        masks: labeled mask array (original size, int32)
    """
    orig_h, orig_w = image_array.shape[:2]
    orig_size = (orig_w, orig_h)
    
    resized = resize_array(image_array, target_size=(segmentation_size, segmentation_size))
    
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


def masks_to_outlines(masks, original_image=None, color=(255, 255, 0), thickness=2):
    """Convert segmentation masks to outline overlay image.
    
    Args:
        masks: labeled mask array from Cellpose
        original_image: optional background image
        color: RGB tuple for outline color (default: yellow)
        thickness: line thickness in pixels
    
    Returns:
        PIL Image with outlines drawn
    """
    # Get outlines using cellpose utility
    outlines = utils.outlines_list(masks)
    
    # Prepare background
    if original_image is None:
        result = np.zeros((*masks.shape, 3), dtype=np.uint8)
    else:
        if original_image.ndim == 2:
            result = np.stack([original_image] * 3, axis=-1)
        else:
            result = original_image.copy()
        
        if result.dtype != np.uint8:
            orig_min, orig_max = result.min(), result.max()
            result = ((result - orig_min) / (orig_max - orig_min) * 255).astype(np.uint8)
    
    # Draw outlines
    for outline in outlines:
        if len(outline) > 0:
            # outline is (N, 2) array of [x, y] coordinates
            pts = outline.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    
    return Image.fromarray(result)


# Distinct colors for multiple masks (up to 4)
MASK_COLORS = [
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Green
]


def draw_multi_mask_outlines(mask_list, original_image=None, thickness=2):
    """Draw outlines for multiple masks with different colors.
    
    Args:
        mask_list: list of labeled mask arrays
        original_image: optional background image
        thickness: line thickness in pixels
    
    Returns:
        PIL Image with all outlines drawn
    """
    if not mask_list:
        if original_image is not None:
            return Image.fromarray(original_image)
        return None
    
    # Prepare background
    first_mask = mask_list[0]
    if original_image is None:
        result = np.zeros((*first_mask.shape, 3), dtype=np.uint8)
    else:
        if original_image.ndim == 2:
            result = np.stack([original_image] * 3, axis=-1)
        else:
            result = original_image.copy()
        
        if result.dtype != np.uint8:
            orig_min, orig_max = result.min(), result.max()
            result = ((result - orig_min) / (orig_max - orig_min) * 255).astype(np.uint8)
    
    # Draw each mask's outlines with a different color
    for i, masks in enumerate(mask_list):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        outlines = utils.outlines_list(masks)
        
        for outline in outlines:
            if len(outline) > 0:
                pts = outline.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    
    return Image.fromarray(result)
