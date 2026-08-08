# **[FireRed-Image-Edit-1.0-Fast](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast)**

FireRed-Image-Edit-1.0-Fast is an experimental, high-performance image editing and style-transfer platform built on top of the `FireRedTeam/FireRed-Image-Edit-1.1` pipeline. The application integrates an optimized transformer architecture (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19`) alongside Flash Attention 3 (`QwenDoubleStreamAttnProcessorFA3`) to execute complex single- and multi-image edit instructions in a rapid 4-step sampling window.

To ensure responsible usage, the pipeline incorporates an integrated NCII (Non-Consensual Intimate Imagery) safety guard model (`hfmlsoc/ncii-light-guard-v01`). The application is served via a single-page web app built with a FastAPI backend server (`gradio.Server`) and a dark red-themed frontend interface featuring a dual-view canvas, A/B comparison slider, history filmstrip, and interactive prompt suggestions.

### **Key Features**

* **NCII Safety Guard Integration:** Evaluates input prompts against a specialized classification guard model (`hfmlsoc/ncii-light-guard-v01`) to detect and block non-consensual intimate imagery requests before execution.
* **Flash Attention 3 (FA3) Acceleration:** Hooks natively into the `QwenDoubleStreamAttnProcessorFA3` processor layer to accelerate cross-attention inference phases while reducing active GPU memory consumption.
* **Multi-Image Reference Manipulation:** Supports uploading multiple reference images to guide edits—such as swapping clothing or accessory items from a reference image onto a target subject while preserving facial identity and background context.
* **Studio SPA Interface:** An interactive single-page application built with modern vanilla web components—featuring an A/B image comparison slider, history filmstrip, quick prompt chips, and drag-and-drop file support.
* **Smart Aspect Ratio Snapping:** Automatically resizes uploaded images to stay within 1024px while snapping width and height to multiples of 8 to prevent shape mismatch errors during inference.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 10.jpg
│   ├── 11.png
│   ├── 2.jpg
│   ├── 3.jpeg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.webp
│   ├── 8.jpg
│   └── 9.png
├── qwenimage/
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── qwen_fa3_processor.py
│   └── transformer_qwenimage.py
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the FireRed-Image-Edit-1.0-Fast environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.12** is required; Python **3.12** or **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for best compatibility.
* **CUDA Version:** CUDA **13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git
cd FireRed-Image-Edit-1.0-Fast
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.22.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
kernels==0.16.0
```

### **Usage**

Once the web server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Asset:** Drag and drop an image into the main canvas workspace, paste an image from your clipboard, or click the upload icon in the left rail.
2. **Refine Instructions:** Type your instructions inside the prompt field, or click one of the **Quick Prompts** chips to instantly fill it. Press ⌘/Ctrl + Enter or click **Edit Image**.
3. **Safety Guard Processing:** Prompts are automatically evaluated by the NCII safety guard model prior to image generation. If a prompt triggers the safety threshold, the request is safely blocked with a warning notification.
4. **Compare & Chain:** Use the **Compare** tool on the left rail to view an A/B slider of the before and after states. Click **Use as Input** to chain multiple edits sequentially.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git](https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast/blob/main/LICENSE.txt)
