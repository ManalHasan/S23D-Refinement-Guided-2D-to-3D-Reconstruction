# Sketch → Refined 2D Image Pipeline

A complete implementation of sketch-guided image refinement using
**SD1.5 + ControlNet**, inspired by the ScribblesAnnotate (IUI '26) paper.
Runs entirely locally on a **16GB GPU** at zero API cost.

---

## Updated Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SKETCH REFINEMENT PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘

  INPUT: Annotated Sketch Image
  (sketch with colored marks: arrows, circles, scribbles, X marks)
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 1: ANNOTATION DETECTION                                     │
│  File: annotation_detector.py      [CPU only, zero VRAM, instant]  │
│                                                                    │
│  1a. Annotation Isolation                                          │
│      ├── If base sketch provided: subtract base from annotated     │
│      └── Else: extract colored pixels (filter out black/white)     │
│                                                                    │
│  1b. Element Detection (OpenCV)                                    │
│      ├── Arrows    → HoughLinesP + contour aspect ratio            │
│      ├── Circles   → HoughCircles                                  │
│      ├── X marks   → crossing line detection                       │
│      └── Scribbles → high perimeter-to-area ratio contours         │
│                                                                    │
│  1c. Color Detection (HSV ranges)                                  │
│      ├── Red    → move / orient                                    │
│      ├── Green  → add new object                                   │
│      ├── Blue   → resize                                           │
│      └── Yellow → pattern / texture                                │
│                                                                    │
│  1d. Action Classification (decision tree)                         │
│      Maps detected elements → 1 of 7 action types:                │
│      [Add object | Change shape | Add pattern |                    │
│       Reference pattern | Change orientation |                     │
│       Change size | Move object]                                   │
│                                                                    │
│  1e. Structured Prompt Generation                                  │
│      Action Guide template + user text → SD-ready prompt           │
│                                                                    │
│  OUTPUT: action_type, confidence, region_mask, structured_prompt   │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SKETCH PREPROCESSING                                     │
│  File: image_generator.py          [CPU, minimal compute]          │
│                                                                    │
│  ├── Resize to 512×512 (SD1.5 native resolution)                   │
│  ├── Convert to RGB                                                │
│  ├── Auto-invert if needed (ensure white bg, black lines)          │
│  ├── Convert to grayscale (strips annotation colors)               │
│  └── Binary threshold (clean up sketch lines)                      │
│                                                                    │
│  OUTPUT: clean control_image (512×512, white bg, black lines)      │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONTROLLED IMAGE GENERATION                              │
│  File: image_generator.py          [GPU, ~6-8GB VRAM]              │
│                                                                    │
│  ControlNet (scribble/softedge/lineart/canny)                      │
│  ├── Reads control_image (sketch structure)                        │
│  ├── controlnet_scale=0.9 (how strictly to follow sketch)          │
│  └── Outputs conditioning features → injected into SD UNet         │
│                                                                    │
│  Stable Diffusion 1.5 (frozen backbone)                            │
│  ├── Input: structured_prompt + negative_prompt                    │
│  ├── Condition: ControlNet features                                │
│  ├── Scheduler: UniPC (20 steps, fast)                             │
│  ├── Guidance scale: 7.5                                           │
│  └── Denoising: 20 steps → refined image                          │
│                                                                    │
│  Optional: Region Mask Blending (inspired by DoodleAssist)         │
│  ├── Generated image blended with original in masked region        │
│  └── Gaussian-smoothed mask edges for natural transitions          │
│                                                                    │
│  OUTPUT: refined 2D image(s)                                       │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
                       ▼
  OUTPUT: Refined 2D Image(s) saved to outputs/


VRAM Usage at Each Stage:
  Stage 1: 0 GB    (pure CPU/OpenCV)
  Stage 2: ~0.1 GB (image ops)
  Stage 3: ~6-8 GB (SD1.5 + ControlNet, with cpu_offload)
  Peak:    ~8 GB   (well within 16GB)

Time per image (RTX 3080/4070 16GB, 20 steps):
  Stage 1: ~0.1s
  Stage 2: ~0.1s
  Stage 3: ~3-8s
  Total:   ~4-9s
```

---

## File Structure

```
sketch_pipeline/
├── annotation_detector.py   # Stage 1: OpenCV annotation detection
├── image_generator.py       # Stage 2+3: SD1.5 + ControlNet generation
├── pipeline.py              # Main orchestrator
├── llava_detector.py        # Optional: LLaVA-based detection (replaces OpenCV)
├── run.py                   # CLI entry point
├── requirements.txt         # Dependencies
└── outputs/                 # Generated images saved here
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA (if not already installed)
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. First run downloads model weights automatically (~5GB total)
- SD1.5: ~4GB (cached in ~/.cache/huggingface/)
- ControlNet scribble: ~1.5GB

---

## Usage

### Basic usage
```bash
python run.py --sketch my_sketch.png
```

### With text description
```bash
python run.py --sketch my_sketch.png --text "move the lamp to the right"
```

### With clean base sketch (better detection)
```bash
python run.py --sketch annotated.png --base clean.png --text "add pattern to sofa"
```

### Demo mode (generates test sketch, no files needed)
```bash
python run.py --demo
```

### Manual prompt override (skip detection)
```bash
python run.py --sketch my_sketch.png --prompt "Move the chair to the left side. High quality 2D illustration."
```

### Higher quality output
```bash
python run.py --sketch my_sketch.png --steps 30 --variants 4
```

---

## Python API

```python
from pipeline import SketchRefinementPipeline, PipelineConfig

# Configure
config = PipelineConfig(
    controlnet_type     = "scribble",   # or softedge / lineart / canny
    num_inference_steps = 20,
    guidance_scale      = 7.5,
    controlnet_scale    = 0.9,
    num_variants        = 2,
    seed                = 42,
    output_dir          = "outputs",
)

# Initialize and load
pipeline = SketchRefinementPipeline(config)
pipeline.load()     # loads SD + ControlNet weights (~30s first time)

# Run
result = pipeline.run(
    annotated_sketch_path = "my_sketch.png",
    base_sketch_path      = None,       # optional clean sketch
    user_text             = "move the lamp to the right side",
)

# Use results
result.show_summary()
for i, img in enumerate(result.output_images):
    img.save(f"my_output_{i}.png")

# Free VRAM when done
pipeline.unload()
```

---

## Annotation Guide (Color Conventions)

Draw annotations on your sketch using these colors for best detection:

| Color  | Meaning                     | Example                          |
|--------|-----------------------------|----------------------------------|
| RED    | Move / Change orientation   | Red arrow showing new position   |
| GREEN  | Add new object              | Green scribble where to add      |
| BLUE   | Resize object               | Blue circle around object        |
| YELLOW | Add / Reference pattern     | Yellow scribble on surface       |
| X mark | Remove / Delete object      | Any color X drawn over object    |

---

## Optional: Use LLaVA Instead of OpenCV

For complex or free-form annotations, replace the OpenCV detector with LLaVA:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLaVA model
ollama pull llava

# Run Ollama server
ollama serve &
```

Then in your code:
```python
from llava_detector import LLaVAAnnotationDetector
from image_generator import SketchToImageGenerator
import numpy as np
from PIL import Image

# Step 1: Detect with LLaVA (uses ~10GB VRAM via Ollama)
detector = LLaVAAnnotationDetector(user_text="move the lamp right")
sketch_np = np.array(Image.open("sketch.png"))
detection = detector.detect(sketch_np)
# LLaVA unloads automatically from Ollama memory

# Step 2: Generate with SD (uses ~6-8GB VRAM)
generator = SketchToImageGenerator()
generator.load()
images = generator.generate(
    sketch_image = Image.open("sketch.png"),
    prompt       = detection.structured_prompt,
)
generator.unload()
```

---

## Hyperparameter Tuning Guide

| Parameter           | Lower value          | Higher value              | Recommended |
|---------------------|----------------------|---------------------------|-------------|
| `num_steps`         | Faster, less detail  | Slower, more detail        | 20-30       |
| `guidance_scale`    | More creative        | Strictly follows prompt    | 7.0-9.0     |
| `controlnet_scale`  | Ignores sketch more  | Follows sketch strictly    | 0.7-1.0     |

**If output ignores sketch structure:** increase `controlnet_scale` (try 1.0-1.2)
**If output looks too rigid:** decrease `controlnet_scale` (try 0.6-0.7)
**If output ignores prompt:** increase `guidance_scale` (try 9.0-12.0)
**If output looks oversaturated/unnatural:** decrease `guidance_scale` (try 6.0)

---

## Future: 2D → 3D

Once you have refined 2D images, these free models convert them to 3D:

```bash
# TripoSR (fast, ~5GB VRAM)
pip install tsr
python -c "
from tsr.system import TSR
model = TSR.from_pretrained('stabilityai/TripoSR')
# feed your refined 2D image
"

# Zero123++ (multi-view, ~8GB VRAM)
# https://github.com/SUDO-AI-3D/zero123plus
```
