# **[FireRed-Image-Edit-1.0-Fast](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast)**

FireRed-Image-Edit-1.0-Fast is a high-performance, AI-driven image editing application that utilizes advanced diffusers and the QwenImageEditPlusPipeline for precise, prompt-based image modifications. Incorporating rapid Transformer configurations, the application provides an interactive Gradio web interface with a custom Soft Blue theme for an aesthetically pleasing user experience. Users can leverage powerful flow match euler discrete schedulers to seamlessly edit visual content by submitting an original image alongside descriptive textual instructions. The application operates entirely in Python, efficiently utilizing CUDA capabilities for accelerated machine learning computations, and serves as a fast, state-of-the-art solution for automated, text-guided image manipulation without complex manual editing software.

<img width="1723" height="1590" alt="image" src="https://github.com/user-attachments/assets/645ec872-7279-4e20-8ea3-eb0cb9b3672a" />

## Features

* **Advanced Diffusers Pipeline:** Utilizes the QwenImageEditPlusPipeline integrated with FlowMatchEulerDiscreteScheduler for high-fidelity image editing based on user prompts.
* **Rapid AI Architecture:** Employs optimized transformer structures designed for fast inference, providing quick iterations and real-time responsiveness.
* **Custom Themed Interface:** Provides an interactive, user-friendly Gradio web interface styled with a custom Soft OrangeRed theme for an optimal visual layout.
* **Hardware Acceleration:** Automatically identifies and leverages CUDA-compatible devices for optimal computational performance, rendering complex edits rapidly.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16; falls back to CPU).
- Stable internet for initial model/LoRA downloads.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git
cd FireRed-Image-Edit-1.0-Fast
```

### 2. Install Pre-requirements

Certain system-level or structural dependencies must be configured before setting up the main python environment:

```bash
pip install -r pre-requirements.txt
```

### 3. Install Standard Dependencies

Install the core Python packages, which include critical modules like Diffusers, Accelerate, PEFT, and Gradio:

```bash
pip install -r requirements.txt
```

## How to Run

To start the application and load the local server, run the main Python script:

```bash
python app.py
```

---

### **Running with uv (Recommended)**

`uv` is an ultra-fast Python package installer and dependency resolver. It isolates execution contexts instantly and securely.

**1. Install `uv`**

* **Linux / macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows (PowerShell):** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**2. Clone and Synchronize the Workspace**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/FireRed-Image-Edit-1.0-Fast.git
cd FireRed-Image-Edit-1.0-Fast
uv sync

```

**3. Launch the Web Interface**

```bash
uv run app.py

```

---

Once the model weights are successfully loaded into your device's memory and the server starts, the terminal will provide a local URL (typically `http://127.0.0.1:7860`). Open this link in your web browser to interact with the visual interface.

## Project Structure

* `app.py`: The main entry point script containing the custom Gradio interface setup, pipeline initialization, and inference logic.
* `qwenimage/`: Core directory housing the transformer and processor modules crucial for the underlying image manipulation techniques.
* `requirements.txt`: The primary file listing Python library requirements needed to operate the application correctly.
* `pre-requirements.txt`: A list containing earlier or auxiliary dependency specifications.
* `examples/`: Directory dedicated to storing sample images and expected outputs to verify application functionality.
* `LICENSE.txt`: The legal text detailing the licensing constraints and permissions.

## Workflow

1. Navigate to the local server URL provided after executing the application.
2. Upload a source image that you wish to edit into the input module.
3. Provide a clear, detailed text prompt describing the exact modifications you want the AI to perform on the image.
4. The system executes the QwenImageEditPlusPipeline via the underlying rapid transformers to compute the altered visual output.
5. Retrieve and save the edited image directly from the interface.

## License

This project is open-source. For detailed terms and conditions, refer to the included `LICENSE.txt` file within the repository.

## Contributing

Community contributions are encouraged. Please submit an issue for bug reports or create a Pull Request to propose features, optimize inference times, or improve the user interface.
