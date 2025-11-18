# **Qwen3-VL-HF-Demo**

This is a Gradio-based demo application for the Qwen3-VL multimodal model. It allows users to perform inference on various media types, including images, videos, PDFs, GIFs, and image captioning. The app supports querying and analyzing content using the powerful Qwen3-VL-30B-A3B-Instruct model from Hugging Face.

 ![Screenshot 1](https://github.com/user-attachments/assets/7cf23b6c-ccca-40f2-8547-0016253e4b44)
<img width="1771" height="1081" alt="Screenshot 2025-10-12 at 19-44-46 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/3cb7d6ee-1e9e-4674-8f3d-7ad9cedbceb4" />
<img width="1775" height="1273" alt="Screenshot 2025-10-12 at 19-17-12 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/befe4813-39b1-47aa-984f-3f994483ea2e" />
<img width="1748" height="983" alt="Screenshot 2025-10-12 at 19-50-02 Qwen3 VL HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/afc3b372-059a-4442-a570-1958e5b1693e" />

## Features

- **Image Inference**: Upload an image and query it for analysis, OCR, captioning, or problem-solving.
- **Video Inference**: Upload a video and describe or explain its content in detail.
- **PDF Inference**: Upload a PDF, preview pages with navigation, and query for summarization, extraction, or analysis.
- **GIF Inference**: Upload a GIF and describe its animation or actions.
- **Image Captioning**: Generate detailed captions and attributes for uploaded images.
- **Advanced Options**: Customize generation parameters like max new tokens, temperature, top-p, top-k, and repetition penalty.
- **Examples**: Pre-loaded examples for each tab to get started quickly.

## Requirements

To run this app, install the following dependencies:

```
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
transformers==4.57.0
huggingface_hub
albumentations
qwen-vl-utils
pyvips-binary
sentencepiece
opencv-python
docling-core
python-docx
torchvision
supervision
matplotlib
pdf2image
num2words
reportlab
html2text
xformers
markdown
requests
pymupdf
loguru
hf_xet
spaces
pyvips
pillow
gradio
einops
httpx
click
torch
fpdf
timm
av
```

You can install them using pip:

```bash
pip install -r requirements.txt
```

(Note: Ensure you have CUDA-enabled GPU for optimal performance, as the model uses torch.float16.)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-HF-Demo.git
   cd Qwen3-VL-HF-Demo
   ```

2. Install the requirements (as listed above).

3. Download the model if needed (the script loads it automatically from Hugging Face).

## Usage

Run the app using Python:

```bash
python app.py
```

- The app will launch a Gradio interface in your browser.
- Select a tab (Image, Video, PDF, GIF, or Caption).
- Upload media and enter a query.
- Adjust advanced options if desired.
- Click "Submit" to generate output.
- For PDFs, use navigation buttons to preview pages.

The app supports streaming output for real-time generation and renders results in both raw text and Markdown formats.

## Acknowledgements

App by: [Prithiv Sakthi U R](https://huggingface.co/prithivMLmods)  

Model: [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)  

## License

Apache License: Version 2.0
