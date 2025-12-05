---
title: Cellpose Anywhere
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
short_description: Cellpose segmentation GUI with multi-channel support
---

# Cellpose Anywhere

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/iranovianti/cellpose-anywhere)

A portable Gradio-based GUI for running [Cellpose](#credits) segmentation.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Usage

1. Upload images
2. Select channel(s) for segmentation
3. Click "Run Cellpose"

## Dependencies

See `requirements.txt`

## TODO

- [ ] Make `SEGMENTATION_SIZE` configurable (currently hardcoded to 128×128 for faster processing, but incompatible with DIC images)
- [ ] Add download button for ImageJ ROI files (investigate using `gr.File` instead of Cellpose's built-in `save_rois` zip output)
- [ ] Improve channel display in `selected_image`:
  - Label each channel when `display_mode` is Grayscale
  - Show warning when RGB Stack mode is used with >3 channels (or >4 for RGBA), suggesting Channel Grayscale mode instead

## Credits

This app uses [Cellpose](https://github.com/MouseLand/cellpose) for segmentation.

> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.
