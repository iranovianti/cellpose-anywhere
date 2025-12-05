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

from cpa import read_image_array, normalize_to_uint8, array_to_display_pil
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
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        return None

    h, w, c = arr.shape

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
# Display Functions
# =============================================================================

def show_channel_preview(files, index, channel_selection, channel_combination):
    """Display selected channel(s) as preview."""
    if not files or index is None:
        return None

    try:
        arr = read_image_array(files[index])
        selected = parse_channel_selection(arr, channel_selection, channel_combination)
        return array_to_preview(selected)
    except Exception as e:
        print(f"Error in show_channel_preview: {e}")
        return None


def show_selected_image(files, display_mode, index):
    """Render the selected file according to display mode."""
    if not files or index is None:
        return None

    arr = read_image_array(files[index])

    if arr.ndim == 2:
        if display_mode == "RGB Stack":
            return array_to_display_pil(arr)
        else:
            arr8 = normalize_to_uint8(arr)
            return Image.fromarray(arr8).convert("L")

    if arr.ndim == 3:
        h, w, c = arr.shape
        if display_mode == "RGB Stack":
            return array_to_display_pil(arr)
        else:
            arr8 = normalize_to_uint8(arr)
            comp = np.zeros((h, w * c, 3), dtype=np.uint8)
            for i in range(c):
                comp[:, i * w:(i + 1) * w, :] = arr8[:, :, i:i+1]
            return Image.fromarray(comp)

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


# =============================================================================
# Gradio UI
# =============================================================================

with gr.Blocks() as demo:
    gr.Markdown("## Cellpose Anywhere")

    # State
    selected_index = gr.State(None)
    current_masks = gr.State(None)

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
                # Image preview
                with gr.Column(scale=1):
                    display_mode = gr.Radio(
                        label="Display Mode",
                        choices=["RGB Stack", "Channel Grayscale"],
                        value="RGB Stack",
                    )
                    selected_image = gr.Image(
                        label="Selected Image",
                        interactive=False,
                        height=400,
                    )
                    file_metadata = gr.Dataframe(
                        label="File Metadata",
                        headers=["Property", "Value"],
                        interactive=False,
                        wrap=True,
                    )

                # Segmentation
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Segmentation")
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
                            run_cellpose_btn = gr.Button(
                                "Run Cellpose",
                                variant="primary",
                                scale=1,
                            )

                    segmentation_result = gr.Image(
                        label="Segmentation Result",
                        interactive=False,
                        height=400,
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
            choices = ["Stack", "Custom"] + [f"Channel {i}" for i in range(c)]
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

    # Update channel preview
    channel_selector.change(
        show_channel_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination],
        outputs=segmentation_result,
    )
    channel_combination.submit(
        show_channel_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination],
        outputs=segmentation_result,
    )
    selected_index.change(
        show_channel_preview,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination],
        outputs=segmentation_result,
    )

    # Run segmentation
    run_cellpose_btn.click(
        run_segmentation,
        inputs=[file_uploader, selected_index, channel_selector, channel_combination],
        outputs=[segmentation_result, current_masks],
    )

    # Update main image display
    selected_index.change(
        show_selected_image,
        inputs=[file_uploader, display_mode, selected_index],
        outputs=selected_image,
    )
    display_mode.change(
        show_selected_image,
        inputs=[file_uploader, display_mode, selected_index],
        outputs=selected_image,
    )

    # Update metadata
    selected_index.change(
        get_file_metadata,
        inputs=[file_uploader, selected_index],
        outputs=file_metadata,
    )


if __name__ == "__main__":
    demo.launch()
