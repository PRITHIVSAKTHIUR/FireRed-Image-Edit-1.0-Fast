# FireRed-Image-Edit-1.0-Fast

A Gradio-based web application for performing image editing tasks using the FireRed-Image-Edit-1.0 model with accelerated 4-step inference. Supports single and multi-image editing through natural language prompts.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Interface Guide](#interface-guide)
- [Advanced Settings](#advanced-settings)
- [Editing Capabilities](#editing-capabilities)
- [Multi-Image Editing](#multi-image-editing)
- [Examples](#examples)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [License](#license)
- [Credits](#credits)

---

## Overview

FireRed-Image-Edit-1.0-Fast is an image editing application that leverages the FireRed-Image-Edit-1.0 pipeline with a distilled transformer model for fast inference. Users provide one or more input images along with a natural language prompt describing the desired edit, and the system generates the modified output image. The default configuration uses only 4 inference steps, enabling rapid editing while maintaining quality.

---

## Features

- **Fast Inference** -- 4-step generation by default using a distilled transformer model, significantly reducing processing time compared to the standard pipeline.
- **Natural Language Prompts** -- Describe edits in plain English rather than manipulating parameters or masks.
- **Multi-Image Input** -- Upload multiple reference images for tasks such as clothing transfer, accessory replacement, or style blending.
- **Automatic Dimension Handling** -- Input images are automatically resized to optimal dimensions while preserving aspect ratio.
- **GPU Acceleration** -- Runs on CUDA-enabled GPUs with bfloat16 precision for efficient memory usage.
- **Flash Attention 3** -- Attempts to use Flash Attention 3 processing for improved transformer performance when available.
- **Custom Theme** -- Orange-red themed interface built on Gradio's Soft theme.
- **MCP Server Support** -- Launches with Model Context Protocol server enabled for programmatic access.
- **Memory Management** -- Automatic garbage collection and CUDA cache clearing between inference calls.

---

## Model Details

| Component | Source |
|-----------|--------|
| Base Pipeline | [FireRedTeam/FireRed-Image-Edit-1.0](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0) |
| Distilled Transformer | [prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19) |
| Architecture | QwenImageTransformer2DModel with FlowMatchEulerDiscreteScheduler |
| Precision | bfloat16 |
| Attention Processor | QwenDoubleStreamAttnProcessorFA3 (Flash Attention 3) |

---

## Requirements

### Hardware

- CUDA-capable GPU with sufficient VRAM (recommended 16GB or more)
- CPU fallback is supported but not recommended for practical use

### Software

- Python 3.8 or higher
- PyTorch with CUDA support
- Gradio
- Diffusers
- NumPy
- Pillow

### Python Dependencies

```
torch
gradio
numpy
Pillow
diffusers
spaces
```

Additional dependencies are required from the `qwenimage` module:

- `pipeline_qwenimage_edit_plus` -- Custom pipeline class
- `transformer_qwenimage` -- Custom transformer model class
- `qwen_fa3_processor` -- Flash Attention 3 processor class

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd FireRed-Image-Edit-1.0-Fast
```

### Install Dependencies

```bash
pip install torch torchvision gradio numpy Pillow diffusers
pip install spaces
```

### Model Downloads

The models are automatically downloaded from HuggingFace on first run:

- The base pipeline downloads from `FireRedTeam/FireRed-Image-Edit-1.0`
- The distilled transformer downloads from `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19`

Ensure you have sufficient disk space and network access to HuggingFace.

### Launch

```bash
python app.py
```

The application will start and display a local URL (typically `http://127.0.0.1:7860`).

---

## Usage

### Basic Workflow

1. Upload one or more images using the gallery input.
2. Enter a natural language prompt describing the desired edit.
3. Click **Edit Image**.
4. View the result in the output panel.

### Input Format

- **Images** -- Upload through the gallery component. Supports common image formats (JPEG, PNG, WebP, and others).
- **Prompt** -- A text description of the desired modification. Be specific about what should change and what should remain unchanged.

---

## Interface Guide

### Main Panel (Left)

| Element | Description |
|---------|-------------|
| Upload Images | Gallery component accepting one or more images. Displays in a 2-column grid with preview capability. |
| Edit Prompt | Text input for the natural language editing instruction. Supports up to 2 lines of text. |
| Edit Image | Primary action button that triggers the inference process. |

### Output Panel (Right)

| Element | Description |
|---------|-------------|
| Output Image | Displays the generated result in PNG format. Non-interactive display only. |

---

## Advanced Settings

The advanced settings panel is available but hidden by default. It contains the following parameters:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Seed | 0 to 2,147,483,647 | 0 | Random seed for reproducible generation |
| Randomize Seed | Boolean | True | When enabled, generates a random seed for each run |
| Guidance Scale | 1.0 to 10.0 | 1.0 | Controls how closely the output follows the prompt. Higher values produce stronger adherence to the prompt but may reduce quality. |
| Inference Steps | 1 to 50 | 4 | Number of denoising steps. More steps generally improve quality but increase processing time. |

### Negative Prompt

A fixed negative prompt is applied to all generations:

```
worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers,
extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
```

This negative prompt is not user-configurable through the interface.

---

## Editing Capabilities

The model supports a wide range of image editing operations through natural language prompts:

### Style Transfer

- Transform images into different artistic styles (anime, cartoon, oil painting, and others)
- Apply photographic effects (polaroid, film grain, vignette)
- Convert color modes (black and white, sepia, high contrast)

### Object Modification

- Replace accessories (glasses, hats, jewelry)
- Change clothing on subjects
- Add or remove elements from scenes

### Enhancement

- Upscale image quality
- Adjust lighting conditions
- Improve texture and detail

### Text and Overlay

- Add text elements to images
- Apply watermark-style overlays
- Create framed compositions

---

## Multi-Image Editing

The application supports uploading multiple images simultaneously. This enables reference-based editing where one image serves as the target and additional images provide reference material.

### How It Works

1. Upload the target image (the image to be edited) and one or more reference images to the gallery.
2. Reference the images by their order in the prompt (for example, "image 1", "image 2").
3. Describe how elements from the reference images should be applied to the target.

### Common Multi-Image Use Cases

| Use Case | Typical Prompt Pattern |
|----------|----------------------|
| Clothing Transfer | "Replace the current clothing with the clothing from the reference image 2." |
| Accessory Swap | "Replace her glasses with the new glasses from image 1." |
| Style Reference | "Apply the artistic style from image 2 to image 1." |

### Multi-Image Prompt Guidelines

When writing prompts for multi-image edits, include explicit instructions about:

- Which elements to transfer from which image
- What should remain unchanged (face, pose, background, lighting)
- How the transferred elements should integrate (fabric texture, shadows, proportions)
- Overall quality expectations (seamless, high-quality, realistic)

---

## Examples

The application includes built-in examples demonstrating various editing capabilities:

### Single Image Examples

| Input | Prompt | Effect |
|-------|--------|--------|
| Portrait photo | "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed 'Fire-Edit' preserving realistic texture and details" | Applies a polaroid film effect with text overlay |
| Character image | "Transform the image into a dotted cartoon style" | Converts to dotted cartoon rendering |
| Color photo | "Convert it to black and white" | Grayscale conversion |

### Multi-Image Examples

| Inputs | Prompt | Effect |
|--------|--------|--------|
| Person + Glasses | "Replace her glasses with the new glasses from image 1" | Swaps eyewear from reference |
| Person + Clothing | "Replace the current clothing with the clothing from the reference image 2..." | Transfers outfit from reference image |

---

## Technical Details

### Image Dimension Processing

Input images are automatically resized according to the following logic:

1. The longer dimension is scaled to 1024 pixels.
2. The shorter dimension is scaled proportionally to maintain aspect ratio.
3. Both dimensions are rounded down to the nearest multiple of 8 (required by the model architecture).

```
If width > height:
    new_width = 1024
    new_height = round(1024 * height / width) aligned to 8

If height >= width:
    new_height = 1024
    new_width = round(1024 * width / height) aligned to 8
```

### Pipeline Architecture

The pipeline uses a custom implementation based on the Diffusers library:

- **Scheduler** -- FlowMatchEulerDiscreteScheduler for flow matching based generation
- **Transformer** -- QwenImageTransformer2DModel, a custom 2D transformer architecture
- **Attention** -- QwenDoubleStreamAttnProcessorFA3, implementing Flash Attention 3 with double-stream attention processing
- **Precision** -- bfloat16 throughout the pipeline for memory efficiency

### Memory Management

The application implements aggressive memory management:

- `gc.collect()` is called before and after each inference
- `torch.cuda.empty_cache()` clears the GPU memory cache between runs
- The request queue is limited to 30 concurrent entries

### Startup Sequence

1. Check and log CUDA device availability
2. Load the base pipeline from `FireRedTeam/FireRed-Image-Edit-1.0`
3. Load the distilled transformer from `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19`
4. Attempt to set Flash Attention 3 processor (falls back gracefully if unavailable)
5. Move the full pipeline to the CUDA device
6. Launch the Gradio server with SSR mode disabled

---

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Controls which GPU devices are visible to PyTorch |

### Server Configuration

The Gradio server launches with the following settings:

| Setting | Value | Description |
|---------|-------|-------------|
| `max_size` | 30 | Maximum queue size for concurrent requests |
| `mcp_server` | True | Enables Model Context Protocol server |
| `ssr_mode` | False | Disables server-side rendering |
| `show_error` | True | Displays error messages in the interface |

### Theme

The application uses a custom OrangeRedTheme extending Gradio's Soft theme:

- **Primary hue** -- Gray
- **Secondary hue** -- Custom orange-red color palette
- **Neutral hue** -- Slate
- **Text size** -- Large
- **Primary font** -- Outfit (Google Fonts)
- **Monospace font** -- IBM Plex Mono (Google Fonts)

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Input image too large or insufficient VRAM | Reduce image dimensions or use fewer inference steps |
| Flash Attention 3 warning | FA3 not available on the hardware | The application falls back automatically; no action required |
| No images processed | Uploaded files in unsupported format | Use standard image formats (JPEG, PNG) |
| Slow inference | Running on CPU | Ensure a CUDA-capable GPU is available |
| Queue full | More than 30 pending requests | Wait for current requests to complete |

### Debug Information

The application logs the following at startup:

- `CUDA_VISIBLE_DEVICES` environment variable value
- PyTorch version
- Active compute device (CUDA or CPU)
- Flash Attention 3 processor status

---

## Limitations

- **Fixed Negative Prompt** -- The negative prompt cannot be modified through the user interface.
- **Maximum Queue Size** -- The server accepts a maximum of 30 queued requests simultaneously.
- **Image Size Constraint** -- All images are resized to a maximum of 1024 pixels on their longest side.
- **Dimension Alignment** -- Output dimensions are constrained to multiples of 8.
- **GPU Requirement** -- While CPU fallback exists, practical use requires a CUDA-capable GPU.
- **Single Output** -- The pipeline generates one output image per request regardless of the number of input images.
- **Experimental Status** -- The application is designated as experimental by the developers.

---

## License

This project uses models and components from HuggingFace. Refer to the individual model repositories for their respective licenses:

- [FireRedTeam/FireRed-Image-Edit-1.0](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0)
- [prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19)

---

## Credits

- **Base Model** -- [FireRedTeam](https://huggingface.co/FireRedTeam) for the FireRed-Image-Edit-1.0 pipeline
- **Distilled Transformer** -- [prithivMLmods](https://huggingface.co/prithivMLmods) for the Qwen-Image-Edit-Rapid-AIO-V19 model
- **Framework** -- Built with [Gradio](https://gradio.app) and [Diffusers](https://huggingface.co/docs/diffusers)
