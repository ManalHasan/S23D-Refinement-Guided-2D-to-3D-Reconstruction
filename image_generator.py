"""
image_generator.py
-------------------
Sketch → Refined 2D image using Stable Diffusion 1.5 + ControlNet.

Fits comfortably in 16GB VRAM.
Uses:
  - lllyasviel/control_v11p_sd15_scribble  (for rough sketches)
  - lllyasviel/control_v11p_sd15_softedge  (for cleaner sketches)
  - runwayml/stable-diffusion-v1-5          (backbone)

All weights downloaded automatically from HuggingFace (free).
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import cv2


class SketchToImageGenerator:
    """
    Wraps SD1.5 + ControlNet for sketch-guided image generation.

    Parameters
    ----------
    controlnet_type : "scribble" | "softedge" | "lineart" | "canny"
        Choose based on your sketch style:
        - scribble  : rough, loose sketches (recommended for annotated input)
        - softedge  : semi-clean sketches
        - lineart   : clean line art
        - canny     : precise edge maps
    device : "cuda" | "cpu"
    use_half_precision : bool - use float16 to save VRAM (recommended)
    """

    CONTROLNET_MODELS = {
        "scribble":  "lllyasviel/control_v11p_sd15_scribble",
        "softedge":  "lllyasviel/control_v11p_sd15_softedge",
        "lineart":   "lllyasviel/control_v11p_sd15_lineart",
        "canny":     "lllyasviel/control_v11p_sd15_canny",
    }

    BASE_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(
        self,
        controlnet_type: str = "scribble",
        device: str = "cuda",
        use_half_precision: bool = True,
    ):
        self.controlnet_type    = controlnet_type
        self.device             = device
        self.dtype              = torch.float16 if use_half_precision else torch.float32
        self.pipe               = None          # lazy load

    def load(self):
        """
        Load ControlNet + SD pipeline.
        Call this once before generating — takes ~30 seconds first time
        (downloads ~5GB of weights, cached after first run).
        """
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        print(f"[Generator] Loading ControlNet ({self.controlnet_type})...")
        controlnet = ControlNetModel.from_pretrained(
            self.CONTROLNET_MODELS[self.controlnet_type],
            torch_dtype=self.dtype,
        )

        print("[Generator] Loading SD1.5 backbone...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,        # disable NSFW checker (saves ~1GB)
        )

        # Faster scheduler (20 steps instead of default 50)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Memory optimizations for 16GB GPU
        self.pipe.enable_model_cpu_offload()            # moves unused layers to CPU
        self.pipe.enable_xformers_memory_efficient_attention()  # saves ~2GB VRAM

        print("[Generator] Pipeline ready.")

    def unload(self):
        """Free VRAM after generation (call before loading LLaVA etc.)"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("[Generator] Pipeline unloaded, VRAM freed.")

    # ------------------------------------------------------------------
    # Main generation function
    # ------------------------------------------------------------------
    def generate(
        self,
        sketch_image: Image.Image,
        prompt: str,
        negative_prompt: str = (
            "blurry, low quality, distorted, deformed, "
            "ugly, bad anatomy, watermark, text, signature"
        ),
        region_mask: np.ndarray = None,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.9,
        num_images: int = 1,
        seed: int = None,
    ) -> list[Image.Image]:
        """
        Generate refined 2D image from sketch.

        Parameters
        ----------
        sketch_image     : PIL Image of the sketch (with or without annotations)
        prompt           : structured prompt from annotation detector
        negative_prompt  : what to avoid in generation
        region_mask      : optional numpy binary mask (255=edit, 0=preserve)
                           if provided, blends generated region with original
        num_steps        : denoising steps (20 is good balance of speed/quality)
        guidance_scale   : how strongly to follow prompt (7-9 typical)
        controlnet_scale : how strongly to follow sketch (0.7-1.0)
                           higher = more faithful to sketch structure
        num_images       : how many variants to generate
        seed             : for reproducibility (None = random)

        Returns
        -------
        List of PIL Images
        """
        if self.pipe is None:
            raise RuntimeError("Call .load() before .generate()")

        # --- preprocess sketch ----------------------------------------
        control_image = self._preprocess_sketch(sketch_image)

        # --- set seed -------------------------------------------------
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # --- generate -------------------------------------------------
        print(f"[Generator] Generating {num_images} image(s)...")
        print(f"[Generator] Prompt: {prompt[:80]}...")

        result = self.pipe(
            prompt                      = prompt,
            negative_prompt             = negative_prompt,
            image                       = control_image,
            num_inference_steps         = num_steps,
            guidance_scale              = guidance_scale,
            controlnet_conditioning_scale = controlnet_scale,
            num_images_per_prompt       = num_images,
            generator                   = generator,
        )

        images = result.images

        # --- apply region mask if provided ----------------------------
        if region_mask is not None:
            images = [
                self._apply_region_mask(sketch_image, img, region_mask)
                for img in images
            ]

        return images

    # ------------------------------------------------------------------
    # Sketch preprocessing
    # ------------------------------------------------------------------
    def _preprocess_sketch(self, sketch: Image.Image) -> Image.Image:
        """
        Preprocess sketch for ControlNet input.
        - Resize to 512x512 (SD1.5 native resolution)
        - Convert to RGB
        - For scribble ControlNet: invert if needed (white bg, black lines)
        - Apply mild denoising to clean up annotation marks
        """
        # resize to 512x512
        sketch_resized = sketch.resize((512, 512), Image.LANCZOS)

        # ensure RGB
        if sketch_resized.mode != "RGB":
            sketch_resized = sketch_resized.convert("RGB")

        sketch_np = np.array(sketch_resized)

        # detect if image needs inversion
        # ControlNet scribble expects: white background, black lines
        mean_brightness = sketch_np.mean()
        if mean_brightness < 128:   # dark background
            sketch_np = 255 - sketch_np

        # convert to grayscale and back to RGB
        # (removes color annotations, keeps structure)
        gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)

        # mild thresholding to clean up sketch
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # back to RGB for ControlNet
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    # Region masking (DoodleAssist-inspired)
    # ------------------------------------------------------------------
    def _apply_region_mask(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """
        Blend generated image with original sketch using region mask.
        Inspired by DoodleAssist's regional latent blending concept —
        applied here at pixel level as post-processing.

        mask: binary numpy array (255 = use generated, 0 = use original)
        """
        orig_np = np.array(original.resize((512, 512)).convert("RGB"))
        gen_np  = np.array(generated.convert("RGB"))

        # smooth mask edges for natural blending
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (31, 31), 0)

        # ensure mask has 3 channels
        mask_3ch = np.stack([mask_float] * 3, axis=-1)

        # blend
        blended = (gen_np * mask_3ch + orig_np * (1 - mask_3ch)).astype(np.uint8)
        return Image.fromarray(blended)
