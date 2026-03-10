"""
pipeline.py
-----------
Main orchestrator for the Sketch → Refined 2D Image pipeline.

Full pipeline:
    [Annotated Sketch Input]
           ↓
    [Stage 1: Annotation Detection]   (OpenCV, zero cost, zero VRAM)
    - Detect arrows, circles, X marks, scribbles
    - Detect annotation colors
    - Classify into 7 action types
    - Generate structured SD prompt
           ↓
    [Stage 2: Sketch Preprocessing]   (in-pipeline, minimal compute)
    - Resize to 512x512
    - Normalize: white bg, black lines
    - Strip annotation colors (keep structure only)
    - Build region mask for targeted editing
           ↓
    [Stage 3: SD1.5 + ControlNet Generation]  (16GB GPU)
    - ControlNet: scribble/softedge (sketch structure guidance)
    - SD1.5: photorealistic/illustrated 2D image generation
    - Optional: region mask blending (preserve unedited areas)
           ↓
    [Refined 2D Image Output]
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from annotation_detector import AnnotationDetector, DetectedAnnotation
from image_generator import SketchToImageGenerator


@dataclass
class PipelineConfig:
    # ControlNet type: "scribble" | "softedge" | "lineart" | "canny"
    controlnet_type: str        = "scribble"

    # Generation parameters
    num_inference_steps: int    = 20        # 20 = fast, 30 = higher quality
    guidance_scale: float       = 7.5       # prompt adherence
    controlnet_scale: float     = 0.9       # sketch adherence
    num_variants: int           = 2         # how many outputs to generate
    seed: Optional[int]         = 42        # None for random

    # Output
    output_dir: str             = "outputs"
    save_intermediates: bool    = True      # save detection visualization

    # Negative prompt (what to avoid)
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, ugly, "
        "bad anatomy, watermark, text, signature, noise, "
        "pixelated, jpeg artifacts, oversaturated"
    )

    # Debug
    debug: bool = False


class SketchRefinementPipeline:
    """
    End-to-end pipeline: annotated sketch → refined 2D image.

    Usage
    -----
    pipeline = SketchRefinementPipeline()
    pipeline.load()

    results = pipeline.run(
        annotated_sketch_path = "my_sketch.png",
        user_text             = "move the chair to the left",
    )

    for i, img in enumerate(results.output_images):
        img.save(f"output_{i}.png")

    pipeline.unload()   # free VRAM
    """

    def __init__(self, config: PipelineConfig = None):
        self.config    = config or PipelineConfig()
        self.generator = SketchToImageGenerator(
            controlnet_type    = self.config.controlnet_type,
            use_half_precision = True,
        )
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def load(self):
        """Load SD + ControlNet weights into GPU. Call once before run()."""
        self.generator.load()

    def unload(self):
        """Free VRAM. Call when done."""
        self.generator.unload()

    # ------------------------------------------------------------------
    # Main pipeline run
    # ------------------------------------------------------------------
    def run(
        self,
        annotated_sketch_path: str,
        base_sketch_path: str       = None,
        user_text: str              = "",
        prompt_override: str        = None,
    ) -> "PipelineResult":
        """
        Run the full pipeline.

        Parameters
        ----------
        annotated_sketch_path : path to sketch with colored annotations
        base_sketch_path      : optional path to clean sketch (no annotations)
                                improves annotation isolation
        user_text             : optional typed description from user
                                e.g. "move the dog to the left near the sofa"
        prompt_override       : if provided, skip detection and use this prompt
                                directly (useful for testing)

        Returns
        -------
        PipelineResult
        """
        t_start = time.time()
        print("\n" + "="*60)
        print("  SKETCH REFINEMENT PIPELINE")
        print("="*60)

        # --- load images ----------------------------------------------
        print("\n[Step 1/4] Loading images...")
        annotated = self._load_image(annotated_sketch_path)
        base      = self._load_image(base_sketch_path) if base_sketch_path else None
        print(f"  Sketch loaded: {annotated.size}")

        # --- stage 1: annotation detection ----------------------------
        print("\n[Step 2/4] Detecting annotations...")
        if prompt_override:
            from annotation_detector import ActionType
            detection = DetectedAnnotation(
                action_type      = ActionType.GENERAL_EDIT,
                confidence       = 1.0,
                region_mask      = np.ones(
                    (annotated.size[1], annotated.size[0]),
                    dtype=np.uint8
                ) * 255,
                description      = f"Manual override: {prompt_override}",
                structured_prompt= prompt_override,
            )
        else:
            detector  = AnnotationDetector(
                user_text = user_text,
                debug     = self.config.debug,
            )
            detection = detector.detect(
                np.array(annotated),
                np.array(base) if base else None,
            )

        print(f"  Action:     {detection.action_type.value}")
        print(f"  Confidence: {detection.confidence:.2f}")
        print(f"  Prompt:     {detection.structured_prompt[:80]}...")

        # --- save detection visualization ----------------------------
        if self.config.save_intermediates:
            viz_path = self._save_detection_viz(
                annotated, detection, annotated_sketch_path
            )
        else:
            viz_path = None

        # --- stage 2 + 3: generate refined image ----------------------
        print("\n[Step 3/4] Generating refined 2D image...")
        images = self.generator.generate(
            sketch_image      = annotated,
            prompt            = detection.structured_prompt,
            negative_prompt   = self.config.negative_prompt,
            region_mask       = detection.region_mask,
            num_steps         = self.config.num_inference_steps,
            guidance_scale    = self.config.guidance_scale,
            controlnet_scale  = self.config.controlnet_scale,
            num_images        = self.config.num_variants,
            seed              = self.config.seed,
        )

        # --- save outputs ---------------------------------------------
        print("\n[Step 4/4] Saving outputs...")
        output_paths = []
        stem = Path(annotated_sketch_path).stem
        for i, img in enumerate(images):
            out_path = os.path.join(
                self.config.output_dir,
                f"{stem}_refined_{i+1}.png"
            )
            img.save(out_path)
            output_paths.append(out_path)
            print(f"  Saved: {out_path}")

        t_total = time.time() - t_start
        print(f"\n  Total time: {t_total:.1f}s")
        print("="*60 + "\n")

        return PipelineResult(
            input_path        = annotated_sketch_path,
            detection         = detection,
            output_images     = images,
            output_paths      = output_paths,
            viz_path          = viz_path,
            time_seconds      = t_total,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return img

    def _save_detection_viz(
        self,
        sketch: Image.Image,
        detection: DetectedAnnotation,
        input_path: str,
    ) -> str:
        """Save a visualization showing what was detected."""
        sketch_np = np.array(sketch.resize((512, 512)))

        # overlay region mask
        mask_colored = np.zeros_like(sketch_np)
        mask_norm = detection.region_mask
        if mask_norm.shape[:2] != (512, 512):
            mask_norm = cv2.resize(mask_norm, (512, 512))
        mask_colored[mask_norm > 0] = [0, 255, 0]     # green overlay

        viz = cv2.addWeighted(
            cv2.cvtColor(sketch_np, cv2.COLOR_RGB2BGR), 0.7,
            mask_colored, 0.3,
            0
        )

        # add text label
        label = f"{detection.action_type.value} ({detection.confidence:.0%})"
        cv2.putText(
            viz, label,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2
        )

        stem     = Path(input_path).stem
        viz_path = os.path.join(self.config.output_dir, f"{stem}_detection.png")
        cv2.imwrite(viz_path, viz)
        print(f"  Detection viz saved: {viz_path}")
        return viz_path


@dataclass
class PipelineResult:
    input_path:     str
    detection:      DetectedAnnotation
    output_images:  list
    output_paths:   list
    viz_path:       Optional[str]
    time_seconds:   float

    def show_summary(self):
        print(f"\nPipeline Summary")
        print(f"  Input:      {self.input_path}")
        print(f"  Action:     {self.detection.action_type.value}")
        print(f"  Confidence: {self.detection.confidence:.2f}")
        print(f"  Outputs:    {len(self.output_images)} image(s)")
        for p in self.output_paths:
            print(f"    → {p}")
        print(f"  Time:       {self.time_seconds:.1f}s")
