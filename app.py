"""Cellpose Anywhere.

Run Cellpose segmentation anywhere — HuggingFace, Colab, or locally.
Supports multi-format and multi-channel images with channel selection.
"""

import os

import gradio as gr
import numpy as np
from PIL import Image

# Conditional import for HuggingFace Spaces ZeroGPU
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

from cpa import read_image_array, normalize_to_uint8, array_to_display_pil, save_masks_as_rois
from cpa.image_segmentation import run_cellpose_segmentation, masks_to_overlay


# =============================================================================
# Helper Functions
# =============================================================================

def get_filenames(files):
    """Extract filenames from uploaded files."""
    if not files:
        return []
    return [os.path.basename(f.name) for f in files]


def get_file_metadata(files, index):
    """Extract metadata for a selected file."""
    if not files or index is None:
        return []

    f = files[index]
    filename = os.path.basename(f.name)

    try:
        arr = read_image_array(f)

        if arr.ndim == 2:
            height, width = arr.shape
            channels = 1
        else:
            height, width, channels = arr.shape

        file_size = os.path.getsize(f.name)
        if file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"

        ext = os.path.splitext(f.name)[1].lower()

        return [
            ["Filename", filename],
            ["Format", ext],
            ["Dimensions", f"{width} × {height}"],
            ["Channels", str(channels)],
            ["Data Type", str(arr.dtype)],
            ["File Size", size_str],
        ]
    except Exception as e:
        return [["Error", str(e)]]


# =============================================================================
# Channel Selection Logic
# =============================================================================

def parse_channel_selection(arr, channel_selection, channel_combination):
    """Parse channel selection and return the appropriate array slice.
    
    Returns:
        numpy array or None if invalid selection
        For 'All (Grayscale)', returns a composite side-by-side image
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        return None

    h, w, c = arr.shape

    if channel_selection == "All (Grayscale)":
        # Create side-by-side grayscale composite
        arr8 = normalize_to_uint8(arr)
        comp = np.zeros((h, w * c), dtype=np.uint8)
        for i in range(c):
            comp[:, i * w:(i + 1) * w] = arr8[:, :, i]
        return comp

    if channel_selection == "Stack":
        return arr

    if channel_selection == "Custom":
        if not channel_combination or not channel_combination.strip():
            return None
        try:
            channel_nums = [int(x.strip()) for x in channel_combination.split(',')]
            if not all(0 <= ch < c for ch in channel_nums):
                return None
            channel_nums = sorted(set(channel_nums))

            if len(channel_nums) == 1:
                return arr[:, :, channel_nums[0]]
            else:
                rgb_stack = np.zeros((h, w, 3), dtype=arr.dtype)
                for i, ch_num in enumerate(channel_nums[:3]):
                    rgb_stack[:, :, i] = arr[:, :, ch_num]
                return rgb_stack
        except (ValueError, IndexError):
            return None

    if channel_selection.startswith("Channel "):
        try:
            channel_num = int(channel_selection.split(" ")[1])
            if 0 <= channel_num < c:
                return arr[:, :, channel_num]
        except (ValueError, IndexError):
            pass

    return None


def array_to_preview(arr):
    """Convert array to PIL Image for preview display."""
    if arr is None:
        return None
    if arr.ndim == 2:
        arr8 = normalize_to_uint8(arr)
        return Image.fromarray(arr8).convert("L")
    return array_to_display_pil(arr)


# =============================================================================
# Segmentation
# =============================================================================

def _run_segmentation_impl(files, index, channel_selection, channel_combination):
    """Run Cellpose segmentation on the selected image/channels."""
    if not files or index is None:
        return None, None

    try:
        arr = read_image_array(files[index])
        img_array = parse_channel_selection(arr, channel_selection, channel_combination)

        if img_array is None:
            return None, None

        masks = run_cellpose_segmentation(img_array)
        overlay = masks_to_overlay(masks, img_array, alpha=0.5)

        # Ensure overlay is RGB for Gradio compatibility
        if isinstance(overlay, Image.Image):
            overlay = overlay.convert("RGB")

        return overlay, masks

    except Exception as e:
        print(f"Error in run_segmentation: {e}")
        return None, None


# Apply @spaces.GPU() decorator only on HuggingFace Spaces
if SPACES_AVAILABLE:
    run_segmentation = spaces.GPU()(_run_segmentation_impl)
else:
    run_segmentation = _run_segmentation_impl


def download_rois(masks, files, index, mask_number=1):
    """Save masks as ImageJ ROI zip file for download."""
    if masks is None or not files or index is None:
        return None
    
    try:
        filename_base = os.path.splitext(os.path.basename(files[index].name))[0]
        roi_path = save_masks_as_rois(masks, filename_base, mask_number)
        return roi_path
    except Exception as e:
        print(f"Error saving ROIs: {e}")
        return None


# =============================================================================
# Gradio UI
# =============================================================================

with gr.Blocks() as demo:
    gr.Markdown("## Cellpose Anywhere")

    # State
    selected_index = gr.State(None)
    masks_cache = gr.State({})  # Dict mapping filename -> list of {"masks": array, "roi_path": path}
    MAX_MASKS = 4

    with gr.Row():
        # Left column: File upload
        with gr.Column(scale=1):
            file_uploader = gr.File(
                label="Upload Images",
                file_types=["image"],
                file_count="multiple",
            )

        # Right column: Image display and segmentation
        with gr.Column(scale=2):
            file_selector = gr.Dropdown(
                label="Select File",
                choices=[],
                interactive=True,
            )

            with gr.Row():
                # Left: Channel selection and image display
                with gr.Column(scale=2):
                    with gr.Row():
                        channel_selector = gr.Dropdown(
                            label="Channel",
                            choices=["Stack"],
                            value="Stack",
                            interactive=True,
                            scale=2,
                        )
                        channel_combination = gr.Textbox(
                            label="Custom Channels",
                            placeholder="e.g., 0, 1, 2",
                            interactive=False,
                            scale=1,
                        )

                    image_display = gr.Image(
                        label="Image",
                        interactive=False,
                        height=400,
                    )

                # Right: Segmentation controls and download
                with gr.Column(scale=1):
                    run_cellpose_btn = gr.Button(
                        "Run Cellpose",
                        variant="primary",
                    )
                    download_roi_file = gr.File(
                        label="Download ROIs",
                        file_count="multiple",
                    )
                    gr.Markdown("**Display Layers**")
                    show_image_layer = gr.Checkbox(
                        label="Image",
                        value=True,
                        interactive=True,
                    )
                    mask_checkboxes = gr.CheckboxGroup(
                        label="Masks",
                        choices=[],
                        value=[],
                        interactive=True,
                    )

            # Metadata
            file_metadata = gr.Dataframe(
                label="File Metadata",
                headers=["Property", "Value"],
                interactive=False,
                wrap=True,
            )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    # File upload → update dropdown
    file_uploader.change(
        lambda files: gr.update(choices=get_filenames(files)),
        inputs=file_uploader,
        outputs=file_selector,
    )

    # File selection → update index
    def on_file_select(selected_label, files):
        if not selected_label or not files:
            return None
        filenames = get_filenames(files)
        return filenames.index(selected_label) if selected_label in filenames else None

    file_selector.change(
        on_file_select,
        inputs=[file_selector, file_uploader],
        outputs=selected_index,
    )

    # Index change → update channel choices
    def update_channel_choices(files, index):
        if not files or index is None:
            return gr.update(choices=["Stack"], value="Stack")
        arr = read_image_array(files[index])
        if arr.ndim == 3:
            _, _, c = arr.shape
            choices = ["Stack", "All (Grayscale)", "Custom"] + [f"Channel {i}" for i in range(c)]
            return gr.update(choices=choices, value="Stack")
        return gr.update(choices=["Stack"], value="Stack")

    selected_index.change(
        update_channel_choices,
        inputs=[file_uploader, selected_index],
        outputs=channel_selector,
    )

    # Channel selector → toggle custom textbox
    channel_selector.change(
        lambda sel: gr.update(interactive=(sel == "Custom")),
        inputs=channel_selector,
        outputs=channel_combination,
    )

    # Restore cached mask overlay or show channel preview when switching images
    def restore_cached_or_preview(files, index, channel_sel, channel_comb, cache, show_image, selected_masks):
        """Show cached segmentation overlay if available, otherwise show channel preview."""
        if not files or index is None:
            return None, None
        
        filename = os.path.basename(files[index].name)
        cached_list = cache.get(filename, [])
        
        # Collect ROI paths from all masks
        roi_paths = [item["roi_path"] for item in cached_list if item.get("roi_path")]
        
        arr = read_image_array(files[index])
        img_array = parse_channel_selection(arr, channel_sel, channel_comb)
        if img_array is None:
            return None, None
        
        # Disable mask overlay for "All (Grayscale)" mode (shape mismatch)
        if channel_sel == "All (Grayscale)":
            return array_to_preview(img_array), roi_paths if roi_paths else None
        
        # Get selected mask indices (e.g., ["Mask 1", "Mask 3"] -> [0, 2])
        selected_indices = []
        for m in selected_masks:
            try:
                idx = int(m.split()[-1]) - 1  # "Mask 1" -> 0
                if 0 <= idx < len(cached_list):
                    selected_indices.append(idx)
            except (ValueError, IndexError):
                pass
        
        if cached_list and selected_indices:
            # Combine selected masks
            combined_masks = np.zeros_like(cached_list[0]["masks"])
            for idx in selected_indices:
                m = cached_list[idx]["masks"]
                combined_masks = np.where(m > 0, m, combined_masks)
            
            if show_image:
                overlay = masks_to_overlay(combined_masks, img_array, alpha=0.5)
            else:
                overlay = masks_to_overlay(combined_masks, None, alpha=1.0)
            
            if isinstance(overlay, Image.Image):
                overlay = overlay.convert("RGB")
            return overlay, roi_paths if roi_paths else None
        
        # No masks selected or no cache
        if show_image:
            return array_to_preview(img_array), roi_paths if roi_paths else None
        else:
            return None, roi_paths if roi_paths else None

    # Update image display when channel selection changes (with cached mask overlay if available)
    channel_selector.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes],
        outputs=[image_display, download_roi_file],
    )
    channel_combination.submit(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes],
        outputs=[image_display, download_roi_file],
    )

    # Restore cached mask overlay or show channel preview when switching images
    selected_index.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes],
        outputs=[image_display, download_roi_file],
    )

    # Update display when layer checkboxes change
    show_image_layer.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes],
        outputs=[image_display, download_roi_file],
    )
    mask_checkboxes.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes],
        outputs=[image_display, download_roi_file],
    )

    # Run segmentation and generate ROI download
    # Note: Using gr.File with direct path return instead of gr.DownloadButton
    # because gr.update() with DownloadButton didn't trigger downloads properly
    def run_and_cache_masks(files, index, channel_sel, channel_comb, cache, show_image, selected_masks):
        if not files or index is None:
            return None, cache, None, gr.update(), gr.update()
        
        filename = os.path.basename(files[index].name)
        cached_list = cache.get(filename, [])
        
        # Check if max masks reached
        if len(cached_list) >= MAX_MASKS:
            # Return current state without adding new mask
            display, roi = restore_cached_or_preview(files, index, channel_sel, channel_comb, cache, show_image, selected_masks)
            return display, cache, roi, gr.update(), gr.update(interactive=False)
        
        _, masks = run_segmentation(files, index, channel_sel, channel_comb)
        if masks is not None:
            # Determine mask number (1-based)
            mask_number = len(cached_list) + 1
            
            # Cache masks and ROI path by filename (append to list)
            roi_path = download_rois(masks, files, index, mask_number)
            if filename not in cache:
                cache[filename] = []
            cache[filename].append({"masks": masks, "roi_path": roi_path})
            
            # Collect all ROI paths
            all_roi_paths = [item["roi_path"] for item in cache[filename] if item.get("roi_path")]
            
            # Update checkbox choices and select all
            num_masks = len(cache[filename])
            choices = [f"Mask {i+1}" for i in range(num_masks)]
            
            # Get display with new mask selected
            display, _ = restore_cached_or_preview(files, index, channel_sel, channel_comb, cache, show_image, choices)
            
            # Disable button if max reached
            btn_interactive = num_masks < MAX_MASKS
            
            return display, cache, all_roi_paths, gr.update(choices=choices, value=choices), gr.update(interactive=btn_interactive)
        
        return None, cache, None, gr.update(), gr.update()
    
    run_cellpose_btn.click(
        run_and_cache_masks,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes],
        outputs=[image_display, masks_cache, download_roi_file, mask_checkboxes, run_cellpose_btn],
    )

    # Update mask checkboxes when switching images
    def update_mask_checkboxes(files, index, cache):
        """Update mask checkbox choices based on cached masks for current image."""
        if not files or index is None:
            return gr.update(choices=[], value=[]), gr.update(interactive=True)
        
        filename = os.path.basename(files[index].name)
        cached_list = cache.get(filename, [])
        num_masks = len(cached_list)
        choices = [f"Mask {i+1}" for i in range(num_masks)]
        
        return gr.update(choices=choices, value=choices), gr.update(interactive=num_masks < MAX_MASKS)
    
    selected_index.change(
        update_mask_checkboxes,
        inputs=[file_uploader, selected_index, masks_cache],
        outputs=[mask_checkboxes, run_cellpose_btn],
    )

    # Update metadata
    selected_index.change(
        get_file_metadata,
        inputs=[file_uploader, selected_index],
        outputs=file_metadata,
    )


if __name__ == "__main__":
    demo.launch()
