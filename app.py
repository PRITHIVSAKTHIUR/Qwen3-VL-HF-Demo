import os
import gc
import json
import base64
import time
import ast
import re
from io import BytesIO
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TEXT_LENGTH = int(os.getenv("MAX_INPUT_TEXT_LENGTH", "2048"))

ACCENT = "#4927F5"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

CATEGORIES = ["Query", "Caption", "Point", "Detect"]

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", DEVICE)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=DTYPE if torch.cuda.is_available() else torch.float32,
).to(DEVICE).eval()

image_examples = [
    {"category": "Point", "query": "Detect the children who are out of focus and wearing a white T-shirt.", "image": "examples/5.jpg"},
    {"category": "Detect", "query": "Point out the out-of-focus (all) children.", "image": "examples/5.jpg"},
    {"category": "Detect", "query": "Headlight", "image": "examples/4.jpg"},
    {"category": "Point", "query": "Gun", "image": "examples/3.jpg"},
    {"category": "Query", "query": "Count the total number of boats and describe the environment.", "image": "examples/1.jpg"},
    {"category": "Caption", "query": "a brief", "image": "examples/2.jpg"},
]

def pil_to_data_url(img: Image.Image, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{data}"

def file_to_data_url(path):
    if not os.path.exists(path):
        return ""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def make_thumb_b64(path, max_dim=240):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim))
        return pil_to_data_url(img, "JPEG")
    except Exception as e:
        print("Thumbnail error:", e)
        return ""

def b64_to_pil(b64_str):
    if not b64_str:
        return None
    try:
        if b64_str.startswith("data:"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        return None

def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}

def extract_json_payload(text: str):
    parsed = safe_parse_json(text)
    if parsed not in ({}, None, ""):
        return parsed

    candidates = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for candidate in candidates:
        parsed = safe_parse_json(candidate)
        if parsed not in ({}, None, ""):
            return parsed

    candidates = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    for candidate in candidates:
        parsed = safe_parse_json(candidate)
        if parsed not in ({}, None, ""):
            return parsed

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        parsed = safe_parse_json(text[start:end + 1])
        if parsed not in ({}, None, ""):
            return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = safe_parse_json(text[start:end + 1])
        if parsed not in ({}, None, ""):
            return parsed

    return {}

def clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0

def normalize_point_coords(x, y):
    vals = [float(x), float(y)]
    if max(abs(vals[0]), abs(vals[1])) > 1.5:
        vals = [vals[0] / 1000.0, vals[1] / 1000.0]
    return clamp01(vals[0]), clamp01(vals[1])

def normalize_box_coords(xmin, ymin, xmax, ymax):
    vals = [float(xmin), float(ymin), float(xmax), float(ymax)]

    # Qwen often returns 0-1000
    if max(abs(v) for v in vals) > 1.5:
        vals = [v / 1000.0 for v in vals]

    x1, y1, x2, y2 = vals
    x1, x2 = sorted([clamp01(x1), clamp01(x2)])
    y1, y2 = sorted([clamp01(y1), clamp01(y2)])
    return x1, y1, x2, y2

def parse_point_object(obj):
    if not isinstance(obj, dict):
        return None

    if "point_2d" in obj and isinstance(obj["point_2d"], (list, tuple)) and len(obj["point_2d"]) == 2:
        return normalize_point_coords(obj["point_2d"][0], obj["point_2d"][1])

    if "point" in obj and isinstance(obj["point"], (list, tuple)) and len(obj["point"]) == 2:
        return normalize_point_coords(obj["point"][0], obj["point"][1])

    if "points" in obj and isinstance(obj["points"], list):
        result = []
        for p in obj["points"]:
            if isinstance(p, dict) and "x" in p and "y" in p:
                result.append(normalize_point_coords(p["x"], p["y"]))
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                result.append(normalize_point_coords(p[0], p[1]))
        return result if result else None

    if "x" in obj and "y" in obj:
        return normalize_point_coords(obj["x"], obj["y"])

    return None

def parse_bbox_object(obj):
    if not isinstance(obj, dict):
        return []

    found = []

    if "bbox_2d" in obj and isinstance(obj["bbox_2d"], (list, tuple)) and len(obj["bbox_2d"]) == 4:
        found.append(normalize_box_coords(*obj["bbox_2d"]))

    if "box_2d" in obj and isinstance(obj["box_2d"], (list, tuple)) and len(obj["box_2d"]) == 4:
        found.append(normalize_box_coords(*obj["box_2d"]))

    if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) == 4:
        found.append(normalize_box_coords(*obj["bbox"]))

    if "box" in obj and isinstance(obj["box"], (list, tuple)) and len(obj["box"]) == 4:
        found.append(normalize_box_coords(*obj["box"]))

    if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max"]):
        found.append(normalize_box_coords(obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]))

    if all(k in obj for k in ["xmin", "ymin", "xmax", "ymax"]):
        found.append(normalize_box_coords(obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]))

    if all(k in obj for k in ["left", "top", "right", "bottom"]):
        found.append(normalize_box_coords(obj["left"], obj["top"], obj["right"], obj["bottom"]))

    if "objects" in obj and isinstance(obj["objects"], list):
        for sub in obj["objects"]:
            found.extend(parse_bbox_object(sub))

    if "bboxes" in obj and isinstance(obj["bboxes"], list):
        for sub in obj["bboxes"]:
            if isinstance(sub, (list, tuple)) and len(sub) == 4:
                found.append(normalize_box_coords(*sub))
            elif isinstance(sub, dict):
                found.extend(parse_bbox_object(sub))

    if "boxes" in obj and isinstance(obj["boxes"], list):
        for sub in obj["boxes"]:
            if isinstance(sub, (list, tuple)) and len(sub) == 4:
                found.append(normalize_box_coords(*sub))
            elif isinstance(sub, dict):
                found.extend(parse_bbox_object(sub))

    return found

def parse_structured_output(category: str, output_text: str):
    parsed_json = extract_json_payload(output_text)

    if category == "Point":
        points_result = {"points": []}
        items = parsed_json if isinstance(parsed_json, list) else [parsed_json] if isinstance(parsed_json, dict) else []

        for item in items:
            p = parse_point_object(item)
            if p is None:
                continue
            if isinstance(p, list):
                for x, y in p:
                    points_result["points"].append({"x": x, "y": y})
            else:
                x, y = p
                points_result["points"].append({"x": x, "y": y})

        return json.dumps(points_result, indent=2), points_result

    if category == "Detect":
        objects_result = {"objects": []}
        items = parsed_json if isinstance(parsed_json, list) else [parsed_json] if isinstance(parsed_json, dict) else []

        for item in items:
            boxes = parse_bbox_object(item)
            for x1, y1, x2, y2 in boxes:
                objects_result["objects"].append({
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                })

        # Fallback regex extraction if model returned plain text with arrays
        if not objects_result["objects"]:
            patterns = [
                r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
            ]
            for pattern in patterns:
                matches = re.findall(pattern, output_text)
                for m in matches:
                    x1, y1, x2, y2 = normalize_box_coords(*m)
                    objects_result["objects"].append({
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                    })

        return json.dumps(objects_result, indent=2), objects_result

    return output_text, {}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def annotate_image(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image

    image = image.convert("RGB")
    original_width, original_height = image.size

    if "points" in result and result["points"]:
        points_list = []
        for p in result.get("points", []):
            try:
                px = int(clamp01(p["x"]) * original_width)
                py = int(clamp01(p["y"]) * original_height)
                points_list.append([px, py])
            except Exception:
                continue

        if not points_list:
            return image

        points_array = np.array(points_list).reshape(1, -1, 2)
        key_points = sv.KeyPoints(xy=points_array)
        vertex_annotator = sv.VertexAnnotator(radius=6, color=sv.Color.from_hex(ACCENT))
        annotated_image = vertex_annotator.annotate(
            scene=np.array(image.copy()),
            key_points=key_points
        )
        return Image.fromarray(annotated_image)

    if "objects" in result and result["objects"]:
        # Use PIL drawing instead of supervision to ensure boxes always render.
        img = image.copy()
        draw = ImageDraw.Draw(img)
        color = hex_to_rgb(ACCENT)

        for obj in result["objects"]:
            try:
                x1 = int(clamp01(obj["x_min"]) * original_width)
                y1 = int(clamp01(obj["y_min"]) * original_height)
                x2 = int(clamp01(obj["x_max"]) * original_width)
                y2 = int(clamp01(obj["y_max"]) * original_height)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Main rectangle
                for i in range(3):
                    draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color, width=1)

                # Corner accents
                c = 12
                draw.line([(x1, y1), (x1 + c, y1)], fill=color, width=3)
                draw.line([(x1, y1), (x1, y1 + c)], fill=color, width=3)

                draw.line([(x2, y1), (x2 - c, y1)], fill=color, width=3)
                draw.line([(x2, y1), (x2, y1 + c)], fill=color, width=3)

                draw.line([(x1, y2), (x1 + c, y2)], fill=color, width=3)
                draw.line([(x1, y2), (x1, y2 - c)], fill=color, width=3)

                draw.line([(x2, y2), (x2 - c, y2)], fill=color, width=3)
                draw.line([(x2, y2), (x2, y2 - c)], fill=color, width=3)
            except Exception as e:
                print("Draw bbox error:", e)
                continue

        return img

    return image

def annotate_to_b64(image: Image.Image, result: dict):
    try:
        annotated = annotate_image(image.copy(), result)
        return pil_to_data_url(annotated, "JPEG")
    except Exception as e:
        print("Annotation error:", e)
        return pil_to_data_url(image, "JPEG")

def build_example_cards_html():
    cards = ""
    for i, ex in enumerate(image_examples):
        thumb = make_thumb_b64(ex["image"])
        prompt_short = ex["query"][:72] + ("..." if len(ex["query"]) > 72 else "")
        cards += f"""
        <div class="example-card" data-idx="{i}">
            <div class="example-thumb-wrap">
                {"<img src='" + thumb + "' alt=''>" if thumb else "<div class='example-thumb-placeholder'>Preview</div>"}
            </div>
            <div class="example-meta-row">
                <span class="example-badge">{ex["category"]}</span>
            </div>
            <div class="example-prompt-text">{prompt_short}</div>
        </div>
        """
    return cards

EXAMPLE_CARDS_HTML = build_example_cards_html()

def load_example_data(idx_str):
    try:
        idx = int(str(idx_str).strip())
    except Exception:
        return gr.update(value=json.dumps({"status": "error", "message": "Invalid example index"}))

    if idx < 0 or idx >= len(image_examples):
        return gr.update(value=json.dumps({"status": "error", "message": "Example index out of range"}))

    ex = image_examples[idx]
    img_b64 = file_to_data_url(ex["image"])
    if not img_b64:
        return gr.update(value=json.dumps({"status": "error", "message": "Could not load example image"}))

    return gr.update(value=json.dumps({
        "status": "ok",
        "query": ex["query"],
        "image": img_b64,
        "category": ex["category"],
        "name": os.path.basename(ex["image"]),
    }))

def build_task_prompt(category: str, prompt: str):
    if category == "Query":
        return prompt
    elif category == "Caption":
        return f"Provide a {prompt} length caption for the image."
    elif category == "Point":
        return f"Provide 2d point coordinates for {prompt}. Report in JSON format."
    elif category == "Detect":
        return f"Provide bounding box coordinates for {prompt}. Report in JSON format."
    return prompt

def calc_timeout_process(*args, **kwargs):
    gpu_timeout = kwargs.get("gpu_timeout", None)
    if gpu_timeout is None and args:
        gpu_timeout = args[-1]
    try:
        return int(gpu_timeout)
    except Exception:
        return 60

@spaces.GPU(duration=calc_timeout_process, size="xlarge")
def generate_understanding(category, text, image, max_new_tokens, gpu_timeout=60):
    try:
        if image is None:
            yield json.dumps({"status": "error", "text": "[ERROR] Please upload an image.", "annotation": ""})
            return

        if not text or not str(text).strip():
            yield json.dumps({"status": "error", "text": "[ERROR] Please enter your instruction.", "annotation": ""})
            return

        if len(str(text)) > MAX_INPUT_TEXT_LENGTH * 8:
            yield json.dumps({"status": "error", "text": "[ERROR] Prompt is too long. Please shorten your input.", "annotation": ""})
            return

        category = category if category in CATEGORIES else "Query"

        image = image.convert("RGB")
        image.thumbnail((512, 512))

        final_prompt = build_task_prompt(category, text)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": final_prompt},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_error = {"error": None}

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }

        def _run_generation():
            try:
                model.generate(**generation_kwargs)
            except Exception as e:
                generation_error["error"] = e
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = Thread(target=_run_generation, daemon=True)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            time.sleep(0.01)
            yield json.dumps({"status": "stream", "text": buffer, "annotation": ""})

        thread.join(timeout=1.0)

        if generation_error["error"] is not None:
            err = f"[ERROR] Inference failed: {str(generation_error['error'])}"
            yield json.dumps({"status": "error", "text": err if not buffer.strip() else buffer + "\n\n" + err, "annotation": ""})
            return

        if not buffer.strip():
            yield json.dumps({"status": "error", "text": "[ERROR] No output was generated.", "annotation": ""})
            return

        final_text, structured_result = parse_structured_output(category, buffer)

        if category in ["Point", "Detect"]:
            annotation_b64 = annotate_to_b64(image, structured_result)
        else:
            annotation_b64 = pil_to_data_url(image, "JPEG")

        yield json.dumps({
            "status": "done",
            "text": final_text,
            "annotation": annotation_b64
        })

    except Exception as e:
        yield json.dumps({"status": "error", "text": f"[ERROR] {str(e)}", "annotation": ""})
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_understanding(category, text, image_b64, max_new_tokens_v, gpu_timeout_v):
    try:
        image = b64_to_pil(image_b64)
        yield from generate_understanding(
            category=category,
            text=text,
            image=image,
            max_new_tokens=max_new_tokens_v,
            gpu_timeout=gpu_timeout_v,
        )
    except Exception as e:
        yield json.dumps({"status": "error", "text": f"[ERROR] {str(e)}", "annotation": ""})

def noop():
    return None

COMPUTER_SVG = f"""
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path fill="white" d="M4 5.5C4 4.12 5.12 3 6.5 3h11C18.88 3 20 4.12 20 5.5v8C20 14.88 18.88 16 17.5 16H13v2h3a1 1 0 1 1 0 2H8a1 1 0 1 1 0-2h3v-2H6.5C5.12 16 4 14.88 4 13.5v-8Zm2 0v8c0 .28.22.5.5.5h11a.5.5 0 0 0 .5-.5v-8a.5.5 0 0 0-.5-.5h-11a.5.5 0 0 0-.5.5Z"/>
  <path fill="white" d="M8 7h8v1.8H8V7Zm0 3h5.5v1.8H8V10Z"/>
</svg>
"""

UPLOAD_PREVIEW_SVG = f"""
<svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="8" y="14" width="64" height="52" rx="6" fill="none" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="4 3"/>
    <polygon points="12,62 30,40 42,50 54,34 68,62" fill="rgba(73,39,245,0.14)" stroke="{ACCENT}" stroke-width="1.5"/>
    <circle cx="28" cy="30" r="6" fill="rgba(73,39,245,0.2)" stroke="{ACCENT}" stroke-width="1.5"/>
</svg>
"""

ANNOTATION_PLACEHOLDER_SVG = f"""
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" fill="none">
  <rect x="18" y="24" width="84" height="64" rx="10" stroke="{ACCENT}" stroke-width="3" stroke-dasharray="5 4"/>
  <circle cx="45" cy="49" r="7" fill="rgba(73,39,245,0.22)" stroke="{ACCENT}" stroke-width="2"/>
  <path d="M28 80L49 60L62 71L78 52L92 80" fill="rgba(73,39,245,0.12)" stroke="{ACCENT}" stroke-width="2.5" stroke-linejoin="round"/>
  <rect x="46" y="92" width="28" height="5" rx="2.5" fill="{ACCENT}" opacity="0.9"/>
</svg>
"""

COPY_SVG = f"""<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="{ACCENT}" d="M16 1H4C2.9 1 2 1.9 2 3v12h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>"""
SAVE_SVG = f"""<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="{ACCENT}" d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zM7 5h8v4H7V5zm12 14H5v-6h14v6z"/></svg>"""

CATEGORY_TABS_HTML = "".join([
    f'<button class="model-tab{" active" if c == "Query" else ""}" data-category="{c}"><span class="model-tab-label">{c}</span></button>'
    for c in CATEGORIES
])

css = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow-x:hidden}}
body,.gradio-container{{
    background:#0f0f13!important;
    font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    font-size:14px!important;color:#e4e4e7!important;min-height:100vh;overflow-x:hidden;
}}
.dark body,.dark .gradio-container{{background:#0f0f13!important;color:#e4e4e7!important}}
footer{{display:none!important}}
.hidden-input{{display:none!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important}}

#gradio-run-btn,#example-load-btn{{
    position:absolute!important;left:-9999px!important;top:-9999px!important;
    width:1px!important;height:1px!important;opacity:0.01!important;
    pointer-events:none!important;overflow:hidden!important;
}}

.app-shell{{
    background:#18181b;border:1px solid #27272a;border-radius:16px;
    margin:12px auto;max-width:1400px;overflow:hidden;
    box-shadow:0 25px 50px -12px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.03);
}}
.app-header{{
    background:linear-gradient(135deg,#18181b,#1e1e24);border-bottom:1px solid #27272a;
    padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
}}
.app-header-left{{display:flex;align-items:center;gap:12px}}
.app-logo{{
    width:38px;height:38px;background:linear-gradient(135deg,{ACCENT},#6b50ff,#8d78ff);
    border-radius:10px;display:flex;align-items:center;justify-content:center;
    box-shadow:0 4px 12px rgba(73,39,245,.30);
}}
.app-logo svg{{width:22px;height:22px;fill:#fff;flex-shrink:0}}

.app-title{{
    font-size:18px;font-weight:700;background:linear-gradient(135deg,#f5f5f5,#bdbdbd);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.3px;
}}
.app-badge{{
    font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;
    background:rgba(73,39,245,.10);color:#cfc6ff;border:1px solid rgba(73,39,245,.24);letter-spacing:.3px;
}}
.app-badge.fast{{background:rgba(73,39,245,.08);color:#b9acff;border:1px solid rgba(73,39,245,.20)}}

.model-tabs-bar{{
    background:#18181b;border-bottom:1px solid #27272a;padding:10px 16px;
    display:flex;gap:8px;align-items:center;flex-wrap:wrap;
}}
.model-tab{{
    display:inline-flex;align-items:center;justify-content:center;gap:6px;
    min-width:32px;height:34px;background:transparent;border:1px solid #27272a;
    border-radius:999px;cursor:pointer;font-size:12px;font-weight:600;padding:0 12px;
    color:#ffffff!important;transition:all .15s ease;
}}
.model-tab:hover{{background:rgba(73,39,245,.10);border-color:rgba(73,39,245,.35)}}
.model-tab.active{{background:rgba(73,39,245,.16);border-color:{ACCENT};color:#fff!important;box-shadow:0 0 0 2px rgba(73,39,245,.10)}}
.model-tab-label{{font-size:12px;color:#ffffff!important;font-weight:600}}

.app-main-row{{display:flex;gap:0;flex:1;overflow:hidden}}
.app-main-left{{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #27272a}}
.app-main-right{{width:470px;display:flex;flex-direction:column;flex-shrink:0;background:#18181b}}

#image-drop-zone{{
    position:relative;background:#09090b;height:440px;min-height:440px;max-height:440px;
    overflow:hidden;
}}
#image-drop-zone.drag-over{{outline:2px solid {ACCENT};outline-offset:-2px;background:rgba(73,39,245,.04)}}
.upload-prompt-modern{{
    position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    padding:20px;z-index:20;overflow:hidden;
}}
.upload-click-area{{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    cursor:pointer;padding:28px 36px;max-width:92%;max-height:92%;
    border:2px dashed #3f3f46;border-radius:16px;
    background:rgba(73,39,245,.03);transition:all .2s ease;gap:8px;text-align:center;
    overflow:hidden;
}}
.upload-click-area:hover{{background:rgba(73,39,245,.08);border-color:{ACCENT};transform:scale(1.02)}}
.upload-click-area:active{{background:rgba(73,39,245,.12);transform:scale(.99)}}
.upload-click-area svg{{width:86px;height:86px;max-width:100%;flex-shrink:0}}
.upload-main-text{{color:#a1a1aa;font-size:14px;font-weight:600;margin-top:4px}}
.upload-sub-text{{color:#71717a;font-size:12px}}

.single-preview-wrap{{
    width:100%;height:100%;display:none;align-items:center;justify-content:center;padding:16px;
    overflow:hidden;
}}
.single-preview-card{{
    width:100%;height:100%;max-width:100%;max-height:100%;border-radius:14px;
    overflow:hidden;border:1px solid #27272a;background:#111114;
    display:flex;align-items:center;justify-content:center;position:relative;
}}
.single-preview-card img{{
    width:100%;height:100%;max-width:100%;max-height:100%;
    object-fit:contain;display:block;
}}
.preview-overlay-actions{{
    position:absolute;top:12px;right:12px;display:flex;gap:8px;z-index:5;
}}
.preview-action-btn{{
    display:inline-flex;align-items:center;justify-content:center;
    min-width:34px;height:34px;padding:0 12px;background:rgba(0,0,0,.65);
    border:1px solid rgba(255,255,255,.14);border-radius:10px;cursor:pointer;
    color:#fff!important;font-size:12px;font-weight:600;transition:all .15s ease;
}}
.preview-action-btn:hover{{background:{ACCENT};border-color:{ACCENT};color:#ffffff!important}}

.hint-bar{{
    background:rgba(73,39,245,.05);border-top:1px solid #27272a;border-bottom:1px solid #27272a;
    padding:10px 20px;font-size:13px;color:#a1a1aa;line-height:1.7;
}}
.hint-bar b{{color:#cbbfff;font-weight:600}}
.hint-bar kbd{{
    display:inline-block;padding:1px 6px;background:#27272a;border:1px solid #3f3f46;
    border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#a1a1aa;
}}

.examples-section{{border-top:1px solid #27272a;padding:12px 16px}}
.examples-title{{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;
    letter-spacing:.8px;margin-bottom:10px;
}}
.examples-scroll{{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}}
.examples-scroll::-webkit-scrollbar{{height:6px}}
.examples-scroll::-webkit-scrollbar-track{{background:#09090b;border-radius:3px}}
.examples-scroll::-webkit-scrollbar-thumb{{background:#27272a;border-radius:3px}}
.examples-scroll::-webkit-scrollbar-thumb:hover{{background:#3f3f46}}
.example-card{{
    flex-shrink:0;width:220px;background:#09090b;border:1px solid #27272a;
    border-radius:10px;overflow:hidden;cursor:pointer;transition:all .2s ease;
}}
.example-card:hover{{border-color:{ACCENT};transform:translateY(-2px);box-shadow:0 4px 12px rgba(73,39,245,.14)}}
.example-card.loading{{opacity:.5;pointer-events:none}}
.example-thumb-wrap{{height:120px;overflow:hidden;background:#18181b}}
.example-thumb-wrap img{{width:100%;height:100%;object-fit:cover}}
.example-thumb-placeholder{{
    width:100%;height:100%;display:flex;align-items:center;justify-content:center;
    background:#18181b;color:#3f3f46;font-size:11px;
}}
.example-meta-row{{padding:6px 10px;display:flex;align-items:center;gap:6px}}
.example-badge{{
    display:inline-flex;padding:2px 7px;background:rgba(73,39,245,.12);border-radius:4px;
    font-size:10px;font-weight:600;color:#cbbfff;font-family:'JetBrains Mono',monospace;white-space:nowrap;
}}
.example-prompt-text{{
    padding:0 10px 8px;font-size:11px;color:#a1a1aa;line-height:1.4;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}}

.panel-card{{border-bottom:1px solid #27272a}}
.panel-card-title{{
    padding:12px 20px;font-size:12px;font-weight:600;color:#71717a;
    text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
}}
.panel-card-body{{padding:16px 20px;display:flex;flex-direction:column;gap:8px}}
.modern-label{{font-size:13px;font-weight:500;color:#a1a1aa;margin-bottom:4px;display:block}}
.modern-textarea{{
    width:100%;background:#09090b;border:1px solid #27272a;border-radius:8px;
    padding:10px 14px;font-family:'Inter',sans-serif;font-size:14px;color:#e4e4e7;
    resize:none;outline:none;min-height:100px;transition:border-color .2s;
}}
.modern-textarea:focus{{border-color:{ACCENT};box-shadow:0 0 0 3px rgba(73,39,245,.16)}}
.modern-textarea::placeholder{{color:#3f3f46}}
.modern-textarea.error-flash{{
    border-color:#ef4444!important;box-shadow:0 0 0 3px rgba(239,68,68,.2)!important;animation:shake .4s ease;
}}
@keyframes shake{{0%,100%{{transform:translateX(0)}}20%,60%{{transform:translateX(-4px)}}40%,80%{{transform:translateX(4px)}}}}

.toast-notification{{
    position:fixed;top:24px;left:50%;transform:translateX(-50%) translateY(-120%);
    z-index:9999;padding:10px 24px;border-radius:10px;font-family:'Inter',sans-serif;
    font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px;
    box-shadow:0 8px 24px rgba(0,0,0,.5);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .35s ease;opacity:0;pointer-events:none;
}}
.toast-notification.visible{{transform:translateX(-50%) translateY(0);opacity:1;pointer-events:auto}}
.toast-notification.error{{background:linear-gradient(135deg,#dc2626,#b91c1c);color:#fff;border:1px solid rgba(255,255,255,.15)}}
.toast-notification.warning{{background:linear-gradient(135deg,{ACCENT},#3217bf);color:#fff;border:1px solid rgba(255,255,255,.15)}}
.toast-notification.info{{background:linear-gradient(135deg,#6b50ff,{ACCENT});color:#fff;border:1px solid rgba(255,255,255,.15)}}
.toast-notification .toast-icon{{font-size:16px;line-height:1}}
.toast-notification .toast-text{{line-height:1.3}}

.btn-run{{
    display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
    background:linear-gradient(135deg,{ACCENT},#5f43ff);border:none;border-radius:10px;
    padding:12px 24px;cursor:pointer;font-size:15px;font-weight:600;font-family:'Inter',sans-serif;
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
    transition:all .2s ease;letter-spacing:-.2px;
    box-shadow:0 4px 16px rgba(73,39,245,.30),inset 0 1px 0 rgba(255,255,255,.18);
}}
.btn-run:hover{{
    background:linear-gradient(135deg,#7b64ff,{ACCENT});transform:translateY(-1px);
    box-shadow:0 6px 24px rgba(73,39,245,.38),inset 0 1px 0 rgba(255,255,255,.22);
}}
.btn-run:active{{transform:translateY(0);box-shadow:0 2px 8px rgba(73,39,245,.28)}}
#custom-run-btn,#custom-run-btn *,#run-btn-label,.btn-run,.btn-run *{{
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;fill:#ffffff!important;
}}

.annot-frame{{border-bottom:1px solid #27272a;display:flex;flex-direction:column;position:relative}}
.annot-title{{
    padding:10px 20px;font-size:13px;font-weight:700;text-transform:uppercase;
    letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);color:#fff
}}
.annot-body{{
    background:#09090b;height:320px;display:flex;align-items:center;justify-content:center;
    padding:12px;position:relative;overflow:hidden;
}}
.annot-body img{{
    max-width:100%;max-height:100%;object-fit:contain;border:1px solid #27272a;
    border-radius:10px;background:#111114;display:none;position:relative;z-index:2;
}}
.annot-placeholder{{
    position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:10px;color:#666;z-index:1;padding:16px;text-align:center;
}}
.annot-placeholder svg{{width:92px;height:92px;max-width:100%;opacity:.95}}
.annot-placeholder-title{{font-size:13px;font-weight:600;color:#9b8fff}}
.annot-placeholder-sub{{font-size:12px;color:#666;max-width:260px;line-height:1.5}}

.output-frame{{border-bottom:1px solid #27272a;display:flex;flex-direction:column;position:relative}}
.output-frame .out-title,
.output-frame .out-title *,
#output-title-label{{
    color:#ffffff!important;
    -webkit-text-fill-color:#ffffff!important;
}}
.output-frame .out-title{{
    padding:10px 20px;font-size:13px;font-weight:700;
    text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
    display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;
}}
.out-title-right{{display:flex;gap:8px;align-items:center}}
.out-action-btn{{
    display:inline-flex;align-items:center;justify-content:center;background:rgba(73,39,245,.10);
    border:1px solid rgba(73,39,245,.2);border-radius:6px;cursor:pointer;padding:3px 10px;
    font-size:11px;font-weight:500;color:#cbbfff!important;gap:4px;height:24px;transition:all .15s;
}}
.out-action-btn:hover{{background:rgba(73,39,245,.2);border-color:rgba(73,39,245,.35);color:#ffffff!important}}
.out-action-btn svg{{width:12px;height:12px;fill:{ACCENT}}}
.output-frame .out-body{{
    flex:1;background:#09090b;display:flex;align-items:stretch;justify-content:stretch;
    overflow:hidden;min-height:320px;position:relative;
}}
.output-scroll-wrap{{width:100%;height:100%;padding:0;overflow:hidden}}
.output-textarea{{
    width:100%;height:320px;min-height:320px;max-height:320px;background:#09090b;color:#e4e4e7;
    border:none;outline:none;padding:16px 18px;font-size:13px;line-height:1.6;
    font-family:'JetBrains Mono',monospace;overflow:auto;resize:none;white-space:pre-wrap;
}}
.output-textarea::placeholder{{color:#52525b}}
.output-textarea.error-flash{{box-shadow:inset 0 0 0 2px rgba(239,68,68,.6)}}

.modern-loader{{
    display:none;position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(9,9,11,.92);
    z-index:15;flex-direction:column;align-items:center;justify-content:center;gap:16px;backdrop-filter:blur(4px);
}}
.modern-loader.active{{display:flex}}
.modern-loader .loader-spinner{{
    width:36px;height:36px;border:3px solid #27272a;border-top-color:{ACCENT};
    border-radius:50%;animation:spin .8s linear infinite;
}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.modern-loader .loader-text{{font-size:13px;color:#a1a1aa;font-weight:500}}
.loader-bar-track{{width:200px;height:4px;background:#27272a;border-radius:2px;overflow:hidden}}
.loader-bar-fill{{
    height:100%;background:linear-gradient(90deg,{ACCENT},#7b64ff,{ACCENT});
    background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite;border-radius:2px;
}}
@keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}

.settings-group{{border:1px solid #27272a;border-radius:10px;margin:12px 16px;padding:0;overflow:hidden}}
.settings-group-title{{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.8px;
    padding:10px 16px;border-bottom:1px solid #27272a;background:rgba(24,24,27,.5);
}}
.settings-group-body{{padding:14px 16px;display:flex;flex-direction:column;gap:12px}}
.slider-row{{display:flex;align-items:center;gap:10px;min-height:28px}}
.slider-row label{{font-size:13px;font-weight:500;color:#a1a1aa;min-width:118px;flex-shrink:0}}
.slider-row input[type="range"]{{
    flex:1;-webkit-appearance:none;appearance:none;height:6px;background:#27272a;
    border-radius:3px;outline:none;min-width:0;
}}
.slider-row input[type="range"]::-webkit-slider-thumb{{
    -webkit-appearance:none;width:16px;height:16px;background:linear-gradient(135deg,{ACCENT},#7056ff);
    border-radius:50%;cursor:pointer;box-shadow:0 2px 6px rgba(73,39,245,.35);transition:transform .15s;
}}
.slider-row input[type="range"]::-webkit-slider-thumb:hover{{transform:scale(1.2)}}
.slider-row input[type="range"]::-moz-range-thumb{{
    width:16px;height:16px;background:linear-gradient(135deg,{ACCENT},#7056ff);
    border-radius:50%;cursor:pointer;border:none;box-shadow:0 2px 6px rgba(73,39,245,.35);
}}
.slider-row .slider-val{{
    min-width:58px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;
    font-weight:500;padding:3px 8px;background:#09090b;border:1px solid #27272a;
    border-radius:6px;color:#a1a1aa;flex-shrink:0;
}}

.app-statusbar{{
    background:#18181b;border-top:1px solid #27272a;padding:6px 20px;
    display:flex;gap:12px;height:34px;align-items:center;font-size:12px;
}}
.app-statusbar .sb-section{{
    padding:0 12px;flex:1;display:flex;align-items:center;font-family:'JetBrains Mono',monospace;
    font-size:12px;color:#52525b;overflow:hidden;white-space:nowrap;
}}
.app-statusbar .sb-section.sb-fixed{{
    flex:0 0 auto;min-width:110px;text-align:center;justify-content:center;
    padding:3px 12px;background:rgba(73,39,245,.08);border-radius:6px;color:#cbbfff;font-weight:500;
}}

.exp-note{{padding:10px 20px;font-size:12px;color:#52525b;border-top:1px solid #27272a;text-align:center}}
.exp-note a{{color:#cbbfff;text-decoration:none}}
.exp-note a:hover{{text-decoration:underline}}

::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:#09090b}}
::-webkit-scrollbar-thumb{{background:#27272a;border-radius:4px}}
::-webkit-scrollbar-thumb:hover{{background:#3f3f46}}

@media(max-width:980px){{
    .app-main-row{{flex-direction:column}}
    .app-main-right{{width:100%}}
    .app-main-left{{border-right:none;border-bottom:1px solid #27272a}}
}}
"""

gallery_js = r"""
() => {
function init() {
    if (window.__qwen3uiInitDone) return;

    const dropZone = document.getElementById('image-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick = document.getElementById('upload-click-area');
    const fileInput = document.getElementById('custom-file-input');
    const previewWrap = document.getElementById('single-preview-wrap');
    const previewImg = document.getElementById('single-preview-img');
    const btnUpload = document.getElementById('preview-upload-btn');
    const btnClear = document.getElementById('preview-clear-btn');
    const promptInput = document.getElementById('custom-query-input');
    const runBtnEl = document.getElementById('custom-run-btn');
    const outputArea = document.getElementById('custom-output-textarea');
    const annotImg = document.getElementById('annotated-output-img');
    const annotPlaceholder = document.getElementById('annotated-output-placeholder');
    const imgStatus = document.getElementById('sb-image-status');

    if (!dropZone || !fileInput || !promptInput || !previewWrap || !previewImg) {
        setTimeout(init, 250);
        return;
    }

    window.__qwen3uiInitDone = true;
    let imageState = null;
    let toastTimer = null;
    let examplePoller = null;
    let lastSeenExamplePayload = null;

    function showToast(message, type) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'toast-notification';
            toast.innerHTML = '<span class="toast-icon"></span><span class="toast-text"></span>';
            document.body.appendChild(toast);
        }
        const icon = toast.querySelector('.toast-icon');
        const text = toast.querySelector('.toast-text');
        toast.className = 'toast-notification ' + (type || 'error');
        if (type === 'warning') icon.textContent = '\u26A0';
        else if (type === 'info') icon.textContent = '\u2139';
        else icon.textContent = '\u2717';
        text.textContent = message;
        if (toastTimer) clearTimeout(toastTimer);
        void toast.offsetWidth;
        toast.classList.add('visible');
        toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
    }

    function showLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.add('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Processing...';
    }

    function hideLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Done';
    }

    function setRunErrorState() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Error';
    }

    window.__showToast = showToast;
    window.__showLoader = showLoader;
    window.__hideLoader = hideLoader;
    window.__setRunErrorState = setRunErrorState;

    function flashPromptError() {
        promptInput.classList.add('error-flash');
        promptInput.focus();
        setTimeout(() => promptInput.classList.remove('error-flash'), 800);
    }

    function flashOutputError() {
        if (!outputArea) return;
        outputArea.classList.add('error-flash');
        setTimeout(() => outputArea.classList.remove('error-flash'), 800);
    }

    function getValueFromContainer(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return '';
        const el = container.querySelector('textarea, input');
        return el ? (el.value || '') : '';
    }

    function setGradioValue(containerId, value) {
        const container = document.getElementById(containerId);
        if (!container) return false;
        const el = container.querySelector('textarea, input');
        if (!el) return false;
        const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        const ns = Object.getOwnPropertyDescriptor(proto, 'value');
        if (ns && ns.set) {
            ns.set.call(el, value);
            el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
            el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            return true;
        }
        return false;
    }

    function syncImageToGradio() {
        setGradioValue('hidden-image-b64', imageState ? imageState.b64 : '');
        const txt = imageState ? '1 image uploaded' : 'No image uploaded';
        if (imgStatus) imgStatus.textContent = txt;
    }

    function syncPromptToGradio() {
        setGradioValue('prompt-gradio-input', promptInput.value);
    }

    function syncCategoryToGradio(name) {
        setGradioValue('hidden-category-name', name);
        const phMap = {
            "Query": "e.g., Count the total number of boats and describe the environment.",
            "Caption": "e.g., short, normal, detailed",
            "Point": "e.g., The gun held by the person.",
            "Detect": "e.g., The headlight of the car."
        };
        promptInput.placeholder = phMap[name] || "e.g., describe or detect the object.";
    }

    function updateAnnotationState(src) {
        if (!annotImg || !annotPlaceholder) return;
        if (src) {
            annotImg.src = src;
            annotImg.style.display = 'block';
            annotPlaceholder.style.display = 'none';
        } else {
            annotImg.src = '';
            annotImg.style.display = 'none';
            annotPlaceholder.style.display = 'flex';
        }
    }

    function setPreview(b64, name) {
        imageState = {b64, name: name || 'image'};
        previewImg.src = b64;
        previewWrap.style.display = 'flex';
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        syncImageToGradio();
    }
    window.__setPreview = setPreview;

    function clearPreview() {
        imageState = null;
        previewImg.src = '';
        previewWrap.style.display = 'none';
        if (uploadPrompt) uploadPrompt.style.display = 'flex';
        syncImageToGradio();
        updateAnnotationState('');
    }
    window.__clearPreview = clearPreview;

    function processFile(file) {
        if (!file) return;
        if (!file.type.startsWith('image/')) {
            showToast('Only image files are supported', 'error');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target.result, file.name);
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
        if (file) processFile(file);
        e.target.value = '';
    });

    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload) btnUpload.addEventListener('click', () => fileInput.click());
    if (btnClear) btnClear.addEventListener('click', clearPreview);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
    });

    promptInput.addEventListener('input', syncPromptToGradio);

    function activateCategoryTab(name) {
        document.querySelectorAll('.model-tab[data-category]').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-category') === name);
        });
        syncCategoryToGradio(name);
    }
    window.__activateCategoryTab = activateCategoryTab;

    document.querySelectorAll('.model-tab[data-category]').forEach(btn => {
        btn.addEventListener('click', () => {
            const category = btn.getAttribute('data-category');
            activateCategoryTab(category);
        });
    });

    activateCategoryTab('Query');

    function syncSlider(customId, gradioId) {
        const slider = document.getElementById(customId);
        const valSpan = document.getElementById(customId + '-val');
        if (!slider) return;
        slider.addEventListener('input', () => {
            if (valSpan) valSpan.textContent = slider.value;
            const container = document.getElementById(gradioId);
            if (!container) return;
            container.querySelectorAll('input[type="range"],input[type="number"]').forEach(el => {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, slider.value);
                    el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
    }

    syncSlider('custom-max-new-tokens', 'gradio-max-new-tokens');
    syncSlider('custom-gpu-duration', 'gradio-gpu-duration');

    function validateBeforeRun() {
        const promptVal = promptInput.value.trim();
        if (!imageState && !promptVal) {
            showToast('Please upload an image and enter your instruction', 'error');
            flashPromptError();
            return false;
        }
        if (!imageState) {
            showToast('Please upload an image', 'error');
            return false;
        }
        if (!promptVal) {
            showToast('Please enter your prompt', 'warning');
            flashPromptError();
            return false;
        }
        const currentCategory = (document.querySelector('.model-tab.active') || {}).dataset?.category;
        if (!currentCategory) {
            showToast('Please select a task category', 'error');
            return false;
        }
        return true;
    }

    window.__clickGradioRunBtn = function() {
        if (!validateBeforeRun()) return;
        syncPromptToGradio();
        syncImageToGradio();
        const active = document.querySelector('.model-tab.active');
        if (active) syncCategoryToGradio(active.getAttribute('data-category'));
        if (outputArea) outputArea.value = '';
        updateAnnotationState('');
        showLoader();
        setTimeout(() => {
            const gradioBtn = document.getElementById('gradio-run-btn');
            if (!gradioBtn) {
                setRunErrorState();
                if (outputArea) outputArea.value = '[ERROR] Run button not found.';
                showToast('Run button not found', 'error');
                return;
            }
            const btn = gradioBtn.querySelector('button');
            if (btn) btn.click(); else gradioBtn.click();
        }, 180);
    };

    if (runBtnEl) runBtnEl.addEventListener('click', () => window.__clickGradioRunBtn());

    const copyBtn = document.getElementById('copy-output-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            try {
                const text = outputArea ? outputArea.value : '';
                if (!text.trim()) {
                    showToast('No output to copy', 'warning');
                    flashOutputError();
                    return;
                }
                await navigator.clipboard.writeText(text);
                showToast('Output copied to clipboard', 'info');
            } catch(e) {
                showToast('Copy failed', 'error');
            }
        });
    }

    const saveBtn = document.getElementById('save-output-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            const text = outputArea ? outputArea.value : '';
            if (!text.trim()) {
                showToast('No output to save', 'warning');
                flashOutputError();
                return;
            }
            const blob = new Blob([text], {type: 'text/plain;charset=utf-8'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'qwen3vl_output.txt';
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                URL.revokeObjectURL(a.href);
                document.body.removeChild(a);
            }, 200);
            showToast('Output saved', 'info');
        });
    }

    function applyExamplePayload(raw) {
        try {
            const data = JSON.parse(raw);
            if (data.status === 'ok') {
                if (data.image) setPreview(data.image, data.name || 'example.jpg');
                if (data.query) {
                    promptInput.value = data.query;
                    syncPromptToGradio();
                }
                if (data.category) activateCategoryTab(data.category);
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Example loaded', 'info');
            } else if (data.status === 'error') {
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast(data.message || 'Failed to load example', 'error');
            }
        } catch (e) {
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
        }
    }

    function startExamplePolling() {
        if (examplePoller) clearInterval(examplePoller);
        let attempts = 0;
        examplePoller = setInterval(() => {
            attempts += 1;
            const current = getValueFromContainer('example-result-data');
            if (current && current !== lastSeenExamplePayload) {
                lastSeenExamplePayload = current;
                clearInterval(examplePoller);
                examplePoller = null;
                applyExamplePayload(current);
                return;
            }
            if (attempts >= 100) {
                clearInterval(examplePoller);
                examplePoller = null;
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Example load timed out', 'error');
            }
        }, 120);
    }

    function triggerExampleLoad(idx) {
        const btnWrap = document.getElementById('example-load-btn');
        const btn = btnWrap ? (btnWrap.querySelector('button') || btnWrap) : null;
        if (!btn) return;

        let attempts = 0;

        function writeIdxAndClick() {
            attempts += 1;
            const ok1 = setGradioValue('example-idx-input', String(idx));
            setGradioValue('example-result-data', '');
            const currentVal = getValueFromContainer('example-idx-input');

            if (ok1 && currentVal === String(idx)) {
                btn.click();
                startExamplePolling();
                return;
            }

            if (attempts < 30) {
                setTimeout(writeIdxAndClick, 100);
            } else {
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Failed to initialize example loader', 'error');
            }
        }

        writeIdxAndClick();
    }

    document.querySelectorAll('.example-card[data-idx]').forEach(card => {
        card.addEventListener('click', () => {
            const idx = card.getAttribute('data-idx');
            if (idx === null || idx === undefined || idx === '') return;
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
            card.classList.add('loading');
            showToast('Loading example...', 'info');
            triggerExampleLoad(idx);
        });
    });

    const observerTarget = document.getElementById('example-result-data');
    if (observerTarget) {
        const obs = new MutationObserver(() => {
            const current = getValueFromContainer('example-result-data');
            if (!current || current === lastSeenExamplePayload) return;
            lastSeenExamplePayload = current;
            if (examplePoller) {
                clearInterval(examplePoller);
                examplePoller = null;
            }
            applyExamplePayload(current);
        });
        obs.observe(observerTarget, {childList:true, subtree:true, characterData:true, attributes:true});
    }

    updateAnnotationState('');
    if (outputArea) outputArea.value = '';
    const sb = document.getElementById('sb-run-state');
    if (sb) sb.textContent = 'Ready';
    if (imgStatus) imgStatus.textContent = 'No image uploaded';

    window.__updateAnnotationState = updateAnnotationState;
}
init();
}
"""

wire_outputs_js = r"""
() => {
function watchOutputs() {
    const resultContainer = document.getElementById('gradio-result');
    const outArea = document.getElementById('custom-output-textarea');

    if (!resultContainer || !outArea) { setTimeout(watchOutputs, 500); return; }

    let lastText = '';

    function isErrorText(val) {
        return typeof val === 'string' && val.trim().startsWith('[ERROR]');
    }

    function syncOutput() {
        const el = resultContainer.querySelector('textarea') || resultContainer.querySelector('input');
        if (!el) return;
        const val = el.value || '';

        if (val !== lastText) {
            lastText = val;

            try {
                const data = JSON.parse(val);

                if (data.text !== undefined) {
                    outArea.value = data.text || '';
                    outArea.scrollTop = outArea.scrollHeight;
                }

                if (data.annotation && window.__updateAnnotationState) {
                    window.__updateAnnotationState(data.annotation);
                }

                if (data.status === 'error') {
                    if (window.__setRunErrorState) window.__setRunErrorState();
                    if (window.__showToast) window.__showToast('Inference failed', 'error');
                } else if (data.status === 'done') {
                    if (window.__hideLoader) window.__hideLoader();
                }
            } catch (e) {
                outArea.value = val;
                outArea.scrollTop = outArea.scrollHeight;
                if (val.trim()) {
                    if (isErrorText(val)) {
                        if (window.__setRunErrorState) window.__setRunErrorState();
                        if (window.__showToast) window.__showToast('Inference failed', 'error');
                    } else {
                        if (window.__hideLoader) window.__hideLoader();
                    }
                }
            }
        }
    }

    const observer = new MutationObserver(syncOutput);
    observer.observe(resultContainer, {childList:true, subtree:true, characterData:true, attributes:true});
    setInterval(syncOutput, 500);
}
watchOutputs();
}
"""

with gr.Blocks() as demo:
    hidden_image_b64 = gr.Textbox(value="", elem_id="hidden-image-b64", elem_classes="hidden-input", container=False)
    prompt = gr.Textbox(value="", elem_id="prompt-gradio-input", elem_classes="hidden-input", container=False)
    hidden_category_name = gr.Textbox(value="Query", elem_id="hidden-category-name", elem_classes="hidden-input", container=False)

    max_new_tokens = gr.Slider(
        minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1,
        value=DEFAULT_MAX_NEW_TOKENS,
        elem_id="gradio-max-new-tokens", elem_classes="hidden-input", container=False
    )
    gpu_duration_state = gr.Number(value=60, elem_id="gradio-gpu-duration", elem_classes="hidden-input", container=False)

    result = gr.Textbox(value="", elem_id="gradio-result", elem_classes="hidden-input", container=False)

    example_idx = gr.Textbox(value="", elem_id="example-idx-input", elem_classes="hidden-input", container=False)
    example_result = gr.Textbox(value="", elem_id="example-result-data", elem_classes="hidden-input", container=False)
    example_load_btn = gr.Button("Load Example", elem_id="example-load-btn")

    gr.HTML(f"""
    <div class="app-shell">
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">{COMPUTER_SVG}</div>
                <span class="app-title">Qwen3-VL</span>
                <span class="app-badge">vision enabled</span>
                <span class="app-badge fast">query / caption / point / detect</span>
            </div>
        </div>

        <div class="model-tabs-bar">
            {CATEGORY_TABS_HTML}
        </div>

        <div class="app-main-row">
            <div class="app-main-left">
                <div id="image-drop-zone">
                    <div id="upload-prompt" class="upload-prompt-modern">
                        <div id="upload-click-area" class="upload-click-area">
                            {UPLOAD_PREVIEW_SVG}
                            <span class="upload-main-text">Click or drag an image here</span>
                            <span class="upload-sub-text">Upload one image for multimodal understanding, captioning, point grounding, or object detection</span>
                        </div>
                    </div>

                    <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />

                    <div id="single-preview-wrap" class="single-preview-wrap">
                        <div class="single-preview-card">
                            <img id="single-preview-img" src="" alt="Preview">
                            <div class="preview-overlay-actions">
                                <button id="preview-upload-btn" class="preview-action-btn" title="Replace">Upload</button>
                                <button id="preview-clear-btn" class="preview-action-btn" title="Clear">Clear</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="hint-bar">
                    <b>Upload:</b> Click or drag to add an image &nbsp;&middot;&nbsp;
                    <b>Task:</b> Switch tabs for Query, Caption, Point, or Detect &nbsp;&middot;&nbsp;
                    <kbd>Clear</kbd> removes the current image
                </div>

                <div class="examples-section">
                    <div class="examples-title">Quick Examples</div>
                    <div class="examples-scroll">
                        {EXAMPLE_CARDS_HTML}
                    </div>
                </div>
            </div>

            <div class="app-main-right">
                <div class="panel-card">
                    <div class="panel-card-title">Instruction</div>
                    <div class="panel-card-body">
                        <label class="modern-label" for="custom-query-input">Prompt Input</label>
                        <textarea id="custom-query-input" class="modern-textarea" rows="4" placeholder="e.g., Count the total number of boats and describe the environment."></textarea>
                    </div>
                </div>

                <div style="padding:12px 20px;">
                    <button id="custom-run-btn" class="btn-run">
                        <span id="run-btn-label">Run Understanding</span>
                    </button>
                </div>

                <div class="annot-frame">
                    <div class="annot-title">Annotated Output</div>
                    <div class="annot-body">
                        <div id="annotated-output-placeholder" class="annot-placeholder">
                            {ANNOTATION_PLACEHOLDER_SVG}
                            <div class="annot-placeholder-title">Annotated preview will appear here</div>
                            <div class="annot-placeholder-sub">Point and Detect tasks will render visual overlays after inference. Query and Caption will show the processed image.</div>
                        </div>
                        <img id="annotated-output-img" src="" alt="Annotated output">
                    </div>
                </div>

                <div class="output-frame">
                    <div class="out-title">
                        <span id="output-title-label">Raw Output Stream</span>
                        <div class="out-title-right">
                            <button id="copy-output-btn" class="out-action-btn" title="Copy">{COPY_SVG} Copy</button>
                            <button id="save-output-btn" class="out-action-btn" title="Save">{SAVE_SVG} Save File</button>
                        </div>
                    </div>
                    <div class="out-body">
                        <div class="modern-loader" id="output-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Running understanding...</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="output-scroll-wrap">
                            <textarea id="custom-output-textarea" class="output-textarea" placeholder="Raw output will appear here..." readonly></textarea>
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <div class="settings-group-title">Advanced Settings</div>
                    <div class="settings-group-body">
                        <div class="slider-row">
                            <label>Max new tokens</label>
                            <input type="range" id="custom-max-new-tokens" min="1" max="{MAX_MAX_NEW_TOKENS}" step="1" value="{DEFAULT_MAX_NEW_TOKENS}">
                            <span class="slider-val" id="custom-max-new-tokens-val">{DEFAULT_MAX_NEW_TOKENS}</span>
                        </div>
                        <div class="slider-row">
                            <label>GPU Duration (seconds)</label>
                            <input type="range" id="custom-gpu-duration" min="60" max="300" step="30" value="60">
                            <span class="slider-val" id="custom-gpu-duration-val">60</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="exp-note">
            Multimodal Understanding
        </div>

        <div class="app-statusbar">
            <div class="sb-section" id="sb-image-status">No image uploaded</div>
            <div class="sb-section sb-fixed" id="sb-run-state">Ready</div>
        </div>
    </div>
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=noop, inputs=None, outputs=None, js=gallery_js)
    demo.load(fn=noop, inputs=None, outputs=None, js=wire_outputs_js)

    run_btn.click(
        fn=run_understanding,
        inputs=[
            hidden_category_name,
            prompt,
            hidden_image_b64,
            max_new_tokens,
            gpu_duration_state,
        ],
        outputs=[result],
        js=r"""(c, p, img, mnt, gd) => {
            const categoryEl = document.querySelector('.model-tab.active');
            const category = categoryEl ? categoryEl.getAttribute('data-category') : c;
            const promptEl = document.getElementById('custom-query-input');
            const promptVal = promptEl ? promptEl.value : p;
            const imgContainer = document.getElementById('hidden-image-b64');
            let imgVal = img;
            if (imgContainer) {
                const inner = imgContainer.querySelector('textarea, input');
                if (inner) imgVal = inner.value;
            }
            return [category, promptVal, imgVal, mnt, gd];
        }""",
    )

    example_load_btn.click(
        fn=load_example_data,
        inputs=[example_idx],
        outputs=[example_result],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(
        css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["examples"],
    )