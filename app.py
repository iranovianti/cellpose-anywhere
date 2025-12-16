"""Cellpose Anywhere.

Run Cellpose segmentation anywhere — HuggingFace, Colab, or locally.
Supports multi-format and multi-channel images with channel selection.
"""

import os

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Conditional import for HuggingFace Spaces ZeroGPU
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

from cpa import read_image_array, read_image_stack, normalize_to_uint8, array_to_display_pil, save_masks_as_rois
from cpa.image_segmentation import run_cellpose_segmentation, run_cellpose_segmentation_batch, masks_to_overlay, draw_multi_mask_outlines


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
        frames = read_image_stack(f)
        num_frames = len(frames)
        arr = frames[0]  # Use first frame for dimensions

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

        metadata = [
            ["Filename", filename],
            ["Format", ext],
            ["Dimensions", f"{width} × {height}"],
            ["Channels", str(channels)],
            ["Data Type", str(arr.dtype)],
            ["File Size", size_str],
        ]
        
        # Add frame count if it's a stack
        if num_frames > 1:
            metadata.insert(3, ["Frames", str(num_frames)])
        
        return metadata
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
        # Create side-by-side grayscale composite with channel labels
        arr8 = normalize_to_uint8(arr)
        
        # Add padding at top for labels
        label_height = 35
        comp = np.zeros((h + label_height, w * c), dtype=np.uint8)
        
        # Place each channel side-by-side
        for i in range(c):
            comp[label_height:, i * w:(i + 1) * w] = arr8[:, :, i]
        
        # Convert to PIL to add text labels
        comp_pil = Image.fromarray(comp)
        draw = ImageDraw.Draw(comp_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = ImageFont.load_default()
        
        # Add channel labels centered above each channel (dark bg strip already from zeros)
        for i in range(c):
            label = f"Channel {i}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = i * w + (w - text_width) // 2
            draw.text((x_pos, 6), label, fill=255, font=font)
        
        return np.array(comp_pil)

    if channel_selection == "Stack":
        if c == 2:
            return np.concatenate([arr, np.zeros((h, w, 1), dtype=arr.dtype)], axis=-1)
        elif c > 3:
            # Limit to first 3 channels to avoid PIL treating 4th as alpha
            return arr[:, :, :3]
        else:
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


def parse_channel_selection_batch(frames, channel_selection, channel_combination):
    """Apply channel selection to a list of frames.
    
    Args:
        frames: list of numpy arrays (H, W) or (H, W, C)
        channel_selection: channel selection mode
        channel_combination: custom channel string (for Custom mode)
    
    Returns:
        list of processed numpy arrays
    """
    return [parse_channel_selection(f, channel_selection, channel_combination) for f in frames]


def create_placeholder_image(text, width=400, height=300):
    """Create a placeholder image with centered text."""
    img = Image.new("RGB", (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill=(180, 180, 180), font=font)
    return img


# =============================================================================
# Segmentation
# =============================================================================

def _run_segmentation_impl(files, index, channel_selection, channel_combination, seg_size):
    """Run Cellpose segmentation on the selected image/channels.
    
    For single images, returns (overlay, masks).
    For stacks, returns (list of overlays, list of masks).
    """
    if not files or index is None:
        return None, None

    try:
        frames = read_image_stack(files[index])
        processed_frames = parse_channel_selection_batch(frames, channel_selection, channel_combination)
        
        # Filter out None values (invalid channel selection)
        if any(f is None for f in processed_frames):
            return None, None

        # Run segmentation on all frames
        masks_list = run_cellpose_segmentation_batch(processed_frames, segmentation_size=seg_size)
        
        # Generate overlays for each frame
        overlays = []
        for img_array, masks in zip(processed_frames, masks_list):
            overlay = masks_to_overlay(masks, img_array, alpha=0.5)
            if isinstance(overlay, Image.Image):
                overlay = overlay.convert("RGB")
            overlays.append(overlay)

        # Return single items for single-frame images (backward compatibility)
        if len(frames) == 1:
            return overlays[0], masks_list[0]
        
        return overlays, masks_list

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

with gr.Blocks(theme=gr.themes.Soft(primary_hue="gray", secondary_hue="purple")) as demo:
    gr.Markdown("## Cellpose Anywhere")

    # State
    selected_index = gr.State(None)
    masks_cache = gr.State({})  # Dict mapping filename -> list of {"masks": array, "roi_path": path}
    MAX_MASKS = 4

    # Prepare example images for gallery display
    EXAMPLE_PATHS = ["examples/sample1.tif", "examples/sample2.png", "examples/sample3.png"]
    
    def load_example_thumbnails():
        """Load and convert example images to displayable format."""
        thumbnails = []
        for path in EXAMPLE_PATHS:
            if os.path.exists(path):
                try:
                    frames = read_image_stack(type('File', (), {'name': path})())
                    pil = array_to_display_pil(frames[0])
                    thumbnails.append((pil, os.path.basename(path)))
                except:
                    pass
        return thumbnails
    
    with gr.Row():
        # Left column: File upload
        with gr.Column(scale=1, min_width=200):
            file_uploader = gr.File(
                label="Upload Images",
                file_types=["image"],
                file_count="multiple",
            )
            with gr.Column(visible=True) as examples_container:
                gr.Markdown("**Or try an example:**")
                example_gallery = gr.Gallery(
                    value=load_example_thumbnails(),
                    columns=3,
                    rows=1,
                    height=200,
                    object_fit="cover",
                    allow_preview=False,
                    show_label=False,
                )

        # Right column: Image display and segmentation
        with gr.Column(scale=3):
            file_selector = gr.Dropdown(
                label="Select an image",
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
                    channel_warning = gr.Markdown("", visible=False)

                    image_display = gr.Gallery(
                        label="Image",
                        columns=1,
                        rows=None,
                        height=400,
                        object_fit="contain",
                        allow_preview=True,
                        show_label=True,
                    )

                # Right: Segmentation controls and download
                with gr.Column(scale=1):
                    seg_size_slider = gr.Slider(
                        label="Segmentation Size",
                        minimum=64,
                        maximum=1024,
                        step=64,
                        value=512,
                        info="Larger = slower but better masks",
                    )
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
                    mask_display_mode = gr.Radio(
                        label="Mask Style",
                        choices=["Fill", "Outline"],
                        value="Fill",
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

    # Example gallery click → load to file uploader
    def on_example_select(evt: gr.SelectData):
        # Use the index to get the original path
        if evt.index is not None and evt.index < len(EXAMPLE_PATHS):
            return [EXAMPLE_PATHS[evt.index]]
        return None
    
    example_gallery.select(
        on_example_select,
        outputs=file_uploader,
    )

    # File upload → update dropdown, select last file, and hide examples
    def on_files_upload(files):
        if files:
            filenames = get_filenames(files)
            last_file = filenames[-1]
            last_index = len(filenames) - 1
            return (
                gr.update(choices=filenames, value=last_file),
                gr.update(visible=False),
                last_index,
            )
        return gr.update(choices=[], value=None), gr.update(visible=True), None
    
    file_uploader.change(
        on_files_upload,
        inputs=file_uploader,
        outputs=[file_selector, examples_container, selected_index],
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

    # Index change → update channel choices and warning
    def update_channel_choices(files, index):
        if not files or index is None:
            return gr.update(choices=["Stack"], value="Stack"), gr.update(value="", visible=False)
        frames = read_image_stack(files[index])
        arr = frames[0]  # Use first frame for channel info
        if arr.ndim == 3:
            _, _, c = arr.shape
            choices = ["Stack", "All (Grayscale)", "Custom"] + [f"Channel {i}" for i in range(c)]
            # Show warning if >3 channels and Stack is selected
            if c > 3:
                warning = f"⚠️ Image has {c} channels. Stack mode shows only the first 3."
                return gr.update(choices=choices, value="Stack"), gr.update(value=warning, visible=True)
            return gr.update(choices=choices, value="Stack"), gr.update(value="", visible=False)
        return gr.update(choices=["Stack"], value="Stack"), gr.update(value="", visible=False)

    selected_index.change(
        update_channel_choices,
        inputs=[file_uploader, selected_index],
        outputs=[channel_selector, channel_warning],
    )

    # Channel selector → toggle custom textbox and update warning
    def update_channel_options(files, index, sel):
        custom_interactive = gr.update(interactive=(sel == "Custom"))
        # Check if warning should be shown
        if not files or index is None:
            return custom_interactive, gr.update(value="", visible=False)
        frames = read_image_stack(files[index])
        arr = frames[0]  # Use first frame for channel info
        if arr.ndim == 3:
            _, _, c = arr.shape
            if c > 3 and sel == "Stack":
                warning = f"⚠️ Image has {c} channels. Stack mode shows only the first 3."
                return custom_interactive, gr.update(value=warning, visible=True)
        return custom_interactive, gr.update(value="", visible=False)
    
    channel_selector.change(
        update_channel_options,
        inputs=[file_uploader, selected_index, channel_selector],
        outputs=[channel_combination, channel_warning],
    )

    # Restore cached mask overlay or show channel preview when switching images
    def restore_cached_or_preview(files, index, channel_sel, channel_comb, cache, show_image, selected_masks, display_mode):
        """Show cached segmentation overlay if available, otherwise show channel preview.
        
        Returns a list of (image, label) tuples for Gallery display.
        """
        if not files or index is None:
            return None, None
        
        filename = os.path.basename(files[index].name)
        cached_list = cache.get(filename, [])
        
        # Collect ROI paths from all masks
        roi_paths = [item["roi_path"] for item in cached_list if item.get("roi_path")]
        
        frames = read_image_stack(files[index])
        processed_frames = parse_channel_selection_batch(frames, channel_sel, channel_comb)
        
        # Show placeholder if Custom mode with no channels specified
        if any(f is None for f in processed_frames):
            if channel_sel == "Custom":
                placeholder = create_placeholder_image("Enter channel numbers (e.g., 0, 1, 2)")
                return [(placeholder, "Placeholder")], roi_paths if roi_paths else None
            return None, None
        
        # Disable mask overlay for "All (Grayscale)" mode (shape mismatch)
        if channel_sel == "All (Grayscale)":
            gallery_items = []
            for i, img_array in enumerate(processed_frames):
                preview = array_to_preview(img_array)
                label = f"Frame {i+1}" if len(processed_frames) > 1 else None
                gallery_items.append((preview, label))
            return gallery_items, roi_paths if roi_paths else None
        
        # Get selected mask indices (e.g., ["Mask 1", "Mask 3"] -> [0, 2])
        selected_indices = []
        for m in selected_masks:
            try:
                idx = int(m.split()[-1]) - 1  # "Mask 1" -> 0
                if 0 <= idx < len(cached_list):
                    selected_indices.append(idx)
            except (ValueError, IndexError):
                pass
        
        gallery_items = []
        num_frames = len(processed_frames)
        
        for frame_idx, img_array in enumerate(processed_frames):
            label = f"Frame {frame_idx+1}" if num_frames > 1 else None
            
            if cached_list and selected_indices:
                # Get masks for this frame
                if display_mode == "Outline":
                    mask_list = []
                    for idx in selected_indices:
                        masks_data = cached_list[idx]["masks"]
                        # Handle both single masks and lists of masks (for stacks)
                        if isinstance(masks_data, list):
                            if frame_idx < len(masks_data):
                                mask_list.append(masks_data[frame_idx])
                        else:
                            mask_list.append(masks_data)
                    bg_image = img_array if show_image else None
                    overlay = draw_multi_mask_outlines(mask_list, bg_image)
                else:
                    # Combine selected masks for fill mode
                    first_masks = cached_list[0]["masks"]
                    if isinstance(first_masks, list):
                        if frame_idx < len(first_masks):
                            combined_masks = np.zeros_like(first_masks[frame_idx])
                        else:
                            combined_masks = np.zeros(img_array.shape[:2], dtype=np.int32)
                    else:
                        combined_masks = np.zeros_like(first_masks)
                    
                    for idx in selected_indices:
                        masks_data = cached_list[idx]["masks"]
                        if isinstance(masks_data, list):
                            if frame_idx < len(masks_data):
                                m = masks_data[frame_idx]
                                combined_masks = np.where(m > 0, m, combined_masks)
                        else:
                            combined_masks = np.where(masks_data > 0, masks_data, combined_masks)
                    
                    if show_image:
                        overlay = masks_to_overlay(combined_masks, img_array, alpha=0.5)
                    else:
                        overlay = masks_to_overlay(combined_masks, None, alpha=1.0)
                
                if isinstance(overlay, Image.Image):
                    overlay = overlay.convert("RGB")
                gallery_items.append((overlay, label))
            else:
                # No masks selected or no cache
                if show_image:
                    preview = array_to_preview(img_array)
                    gallery_items.append((preview, label))
                else:
                    gallery_items.append((None, label))
        
        return gallery_items, roi_paths if roi_paths else None

    # Update image display when channel selection changes (with cached mask overlay if available)
    channel_selector.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode],
        outputs=[image_display, download_roi_file],
    )
    channel_combination.submit(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode],
        outputs=[image_display, download_roi_file],
    )

    # Restore cached mask overlay or show channel preview when switching images
    selected_index.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode],
        outputs=[image_display, download_roi_file],
    )

    # Update display when layer checkboxes change
    show_image_layer.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode],
        outputs=[image_display, download_roi_file],
    )
    mask_checkboxes.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode],
        outputs=[image_display, download_roi_file],
    )
    mask_display_mode.change(
        restore_cached_or_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode],
        outputs=[image_display, download_roi_file],
    )

    # Run segmentation and generate ROI download
    # Note: Using gr.File with direct path return instead of gr.DownloadButton
    # because gr.update() with DownloadButton didn't trigger downloads properly
    def run_and_cache_masks(files, index, channel_sel, channel_comb, cache, show_image, selected_masks, display_mode, seg_size):
        if not files or index is None:
            return None, cache, None, gr.update(), gr.update()
        
        filename = os.path.basename(files[index].name)
        cached_list = cache.get(filename, [])
        
        # Check if max masks reached
        if len(cached_list) >= MAX_MASKS:
            # Return current state without adding new mask
            display, roi = restore_cached_or_preview(files, index, channel_sel, channel_comb, cache, show_image, selected_masks, display_mode)
            return display, cache, roi, gr.update(), gr.update(interactive=False)
        
        overlays, masks = run_segmentation(files, index, channel_sel, channel_comb, seg_size)
        if masks is not None:
            # Determine mask number (1-based)
            mask_number = len(cached_list) + 1
            
            # Cache masks by filename (append to list)
            # For stacks, masks will be a list; for single images, it's a single array
            if filename not in cache:
                cache[filename] = []
            cache[filename].append({"masks": masks, "roi_path": None})  # ROI export handled later
            
            # Update checkbox choices and select all
            num_masks = len(cache[filename])
            choices = [f"Mask {i+1}" for i in range(num_masks)]
            
            # Get display with new mask selected
            display, roi = restore_cached_or_preview(files, index, channel_sel, channel_comb, cache, show_image, choices, display_mode)
            
            # Disable button if max reached
            btn_interactive = num_masks < MAX_MASKS
            
            return display, cache, roi, gr.update(choices=choices, value=choices), gr.update(interactive=btn_interactive)
        
        return None, cache, None, gr.update(), gr.update()
    
    run_cellpose_btn.click(
        run_and_cache_masks,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination, masks_cache, show_image_layer, mask_checkboxes, mask_display_mode, seg_size_slider],
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
