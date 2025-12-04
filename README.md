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

## Credits

This app uses [Cellpose](https://github.com/MouseLand/cellpose) for segmentation.

> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.
