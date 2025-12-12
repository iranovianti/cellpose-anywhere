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

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/iranovianti/cellpose-anywhere)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/iranovianti/cellpose-anywhere)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/iranovianti/f8f032b3476f7ec6291b245cef6170ed/cellpose_anywhere.ipynb)

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
- [x] ~~Add download button for ImageJ ROI files~~ (implemented with `gr.File` - works well enough)
- [x] ~~Improve channel display:~~
  - ~~Label each channel when display mode is Grayscale~~
  - ~~Show warning when RGB Stack mode is used with >3 channels~~
- [x] ~~Combine `selected_image` and `segmentation_result` into a single unified display~~
- [x] ~~Add mask overlay options: show/hide toggle + mask vs border/outlines mode~~ (Fill/Outline toggle with multi-color outlines)
- [x] ~~Multi-mask per image support (different channels or parameters)~~ (up to 4 masks per image)
- [x] ~~Add layers checkboxes for toggling visibility of channels and masks~~

## Credits

This app uses [Cellpose](https://github.com/MouseLand/cellpose) for segmentation.

> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.
