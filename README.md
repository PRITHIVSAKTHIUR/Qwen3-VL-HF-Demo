# **Qwen3-VL-HF-Demo**

Qwen3-VL-HF-Demo is an experimental visual reasoning, spatial grounding, and object detection workspace built around Alibaba's state-of-the-art `Qwen/Qwen3-VL-8B-Instruct` multimodal foundation model. The platform establishes four core reasoning workflows: **Query** (free-form VQA), **Caption** (variable-length descriptions), **Point** (2D coordinate keypoint localization), and **Detect** (2D bounding box detection).

The application integrates `supervision` and PIL drawing utilities to dynamically render corner accents, keypoints, and bounding boxes over predicted coordinates (`point_2d` and `bbox_2d`). Featuring a dark-mode frontend with custom JavaScript-driven state management, inline token streaming via `TextIteratorStreamer`, and clipboard/export capabilities, it serves as a lightweight sandbox for evaluating vision-language models.

### **Key Features**

* **Multimodal Spatial Reasoning:** Supports open-ended visual question answering, custom-length captioning, precise 2D keypoint localization (`Point`), and bounding box detection (`Detect`).
* **Dynamic Bounding Box & Point Annotation:** Automatically parses raw model output JSON structures, normalizes coordinates, and overlays visual masks and corner-accented bounding boxes onto the target image.
* **Real-time Token Streaming:** Integrates a threaded generation worker using `TextIteratorStreamer` to display reasoning tokens progressively as they are decoded.
* **Custom Headless UI:** Built using vanilla JavaScript and CSS overlays for drag-and-drop media upload zones, live status bars, category tabs, and toast notifications.
* **Export & Output Management:** Includes one-click tools to copy the generated raw token text to the clipboard or download it as a plain text file.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   └── 5.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To set up the Qwen3-VL-HF-Demo environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.10** is required; Python **3.12** or **3.14** is recommended for optimal performance.
* **PyTorch Version:** `torch==2.11.0` or above is required for best compatibility.
* **CUDA Version:** CUDA **13.0** is recommended (`--extra-index-url https://download.pytorch.org/whl/cu130`), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-HF-Demo.git
cd Qwen3-VL-HF-Demo

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
Upgrade your local package manager:

```bash
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, vision-language utilities, and core libraries listed in `requirements.txt`:

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
gradio==6.20.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
supervision==0.29.1
opencv-python==5.0.0.93

```

### **Usage**

Once the web deployment initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Select Task Category:** Click on one of the top category tabs (**Query**, **Caption**, **Point**, or **Detect**).
2. **Upload Asset:** Drag and drop an image into the upload area (or select one from the **Quick Examples** tray).
3. **Refine Prompt:** Type your instruction inside the prompt field (e.g., *"Detect the children wearing a white T-shirt"* for Detect, or *"Headlight"* for Point).
4. **Execute:** Click **Run Understanding**. The raw token output will stream in real time, and any coordinate predictions will render as an annotated overlay in the output image container.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-HF-Demo.git](https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-HF-Demo.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-HF-Demo](https://huggingface.co/spaces/prithivMLmods/Qwen3-VL-HF-Demo)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-HF-Demo/blob/main/LICENSE)
