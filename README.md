# **FireRed-Image-Edit-1.0-Fast**

FireRed-Image-Edit-1.0-Fast is an experimental, high-performance image manipulation workspace powered by the `FireRedTeam/FireRed-Image-Edit-1.1` pipeline. It utilizes a custom, optimized transformer (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19`) and incorporates Flash Attention 3 (`QwenDoubleStreamAttnProcessorFA3`) to achieve rapid, high-fidelity image edits.

The application goes beyond simple text-to-image prompting by enabling multi-image reference uploads. This allows users to perform complex structural edits—such as seamless clothing swaps, lighting alterations, or face replacements—by passing multiple visual contexts to the model. The backend logic is wrapped in a highly customized, responsive Gradio interface with dynamic JavaScript asset handling, interactive galleries, and a sleek dark theme.

<img width="1723" height="1590" alt="image" src="https://github.com/user-attachments/assets/645ec872-7279-4e20-8ea3-eb0cb9b3672a" />

### **Key Features**

* **Reference-Based Editing:** Upload multiple images to act as context. The model can extract elements (like outfits or glasses) from a reference image and apply them seamlessly to a base image.
* **Flash Attention 3 Integration:** Automatically hooks into `QwenDoubleStreamAttnProcessorFA3` to drastically reduce VRAM overhead and accelerate attention mechanism processing.
* **Custom Headless-Style UI:** Abandons standard Gradio block layouts in favor of a bespoke, terminal-inspired dark theme featuring drag-and-drop zones, live toast notifications, and client-side JavaScript gallery management.
* **Smart Dimension Snapping:** Automatically calculates aspect ratios from uploaded images and scales them to fit within a 1024x1024 bounding box, ensuring all dimensions snap to multiples of 8 to prevent tensor mismatch errors during diffusion.
* **ZeroGPU Compatibility:** Optimized for transient hardware allocation using `spaces.GPU`, including aggressive garbage collection and CUDA cache clearing before and after inference cycles.

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
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To run FireRed-Image-Edit-1.0-Fast locally, configure your environment with the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.12** is needed, Python **3.14** is highly recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal compatibility and Flash Attention 3 support.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It guarantees rapid virtual environment synchronization and reproducible execution paths based on the `uv.lock` file.

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

#### **Standard PIP Installation**

**1. Update Package Manager**
Ensure your local pip is up-to-date:

```bash
pip install pip>=26.1.2

```

**2. Install Dependencies**
Install the core deep learning stack, specific transformer builds, and Gradio framework versions from the requirements file.

```bash
pip install -r requirements.txt

```

### **Usage**

Once the FastAPI web server initializes, open your browser to the local address provided in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Images:** Click or drag images into the upload zone.
* For style transfers or global edits (e.g., "Change to winter"), upload a single image.
* For transfer edits (e.g., "Replace clothing using reference"), upload the base image first, followed by the reference image.


2. **Set Prompt:** Type your edit instructions in the prompt box.
3. **Advanced Settings (Optional):** Expand the settings panel to manually lock the generation seed, adjust the guidance scale, or modify inference steps.
4. **Execute:** Click **Edit Image**. The frontend will display a loading overlay while the GPU processes the diffusion steps, subsequently returning the modified image.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git](https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git)
* **Hugging Face Live Demo:** [https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast/blob/main/LICENSE.txt)
