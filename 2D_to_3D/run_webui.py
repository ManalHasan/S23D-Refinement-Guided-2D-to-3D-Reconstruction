import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

# Add TripoSR to path
ROOT_DIR = Path(__file__).parent
TRIPOSR_DIR = ROOT_DIR / "TripoSR"
if TRIPOSR_DIR.exists():
    sys.path.insert(0, str(TRIPOSR_DIR))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import argparse

# Device logic with "white cast" stability fix
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    # Default to CPU on Mac to avoid the white cast/numerical stability issues
    print("[WebUI] Note: MPS detected but using CPU for stable colors (prevents white cast).")
    device = "cpu"
else:
    device = "cpu"

print(f"[WebUI] Loading model on {device}...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image_pil):
        # Convert to RGBA if not already
        image_pil = image_pil.convert("RGBA")
        arr = np.array(image_pil).astype(np.float32) / 255.0
        # Composite onto 0.5 gray background
        # Handle cases with no alpha channel gracefully
        if arr.shape[-1] == 4:
            rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        else:
            rgb = arr[:, :, :3]
        return Image.fromarray((rgb * 255.0).astype(np.uint8))

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = fill_background(input_image)
    return image


def generate(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = generate(preprocessed, 256, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # TripoSR WebUI
    [TripoSR](https://github.com/VAST-AI-Research/TripoSR) reconstructs 3D models from a single image.
    
    **Tips:**
    1. **Foreground Ratio:** If the model looks squashed or too small, adjust this slider (0.85 is usually best).
    2. **Remove Background:** Only disable this if your image already has a perfect gray/transparent background.
    3. **Mac Users:** This UI is running on CPU to ensure accurate colors and avoid the "white cast" bug.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources=["upload"],
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False,
                )
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Output Model (GLB Format)",
                    interactive=False,
                )
    
    # Example paths need to point to the submodule folder
    example_folder = TRIPOSR_DIR / "examples"
    example_list = [str(f) for f in example_folder.glob("*.png")] + [str(f) for f in example_folder.glob("*.jpeg")]

    if example_list:
        with gr.Row(variant="panel"):
            gr.Examples(
                examples=sorted(example_list),
                inputs=[input_image],
                outputs=[processed_image, output_model_obj, output_model_glb],
                cache_examples=False,
                fn=partial(run_example),
                label="Examples",
                examples_per_page=20,
            )
            
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution],
        outputs=[output_model_obj, output_model_glb],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
    parser.add_argument("--share", action='store_true', help="create a public link")
    args = parser.parse_args()
    
    interface.queue(max_size=1)
    interface.launch(
        share=args.share,
        server_name="0.0.0.0" if args.listen else None, 
        server_port=args.port
    )
