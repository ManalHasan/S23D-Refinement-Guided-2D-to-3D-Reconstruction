"""
run.py
------
Entry point for the Sketch Refinement Pipeline.

Run:
    python run.py --sketch my_sketch.png --text "move the chair left"
    python run.py --sketch my_sketch.png --base clean_sketch.png
    python run.py --demo   (generates a test sketch and runs pipeline)
"""

import argparse
import sys
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def generate_demo_sketch(save_path: str = "demo_sketch.png") -> str:
    """
    Generate a simple annotated demo sketch for testing the pipeline
    without needing a real sketch file.

    Draws:
    - A simple room scene (sofa, table, lamp) in black
    - A red arrow annotation indicating 'move lamp to right'
    """
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # --- sketch lines (black) ---
    # floor line
    draw.line([(50, 380), (460, 380)], fill=(0, 0, 0), width=2)
    # wall line
    draw.line([(50, 100), (50, 380)], fill=(0, 0, 0), width=2)
    draw.line([(460, 100), (460, 380)], fill=(0, 0, 0), width=2)

    # sofa (left side)
    draw.rectangle([(60, 280), (220, 380)], outline=(0, 0, 0), width=2)
    draw.rectangle([(60, 250), (220, 290)], outline=(0, 0, 0), width=2)
    # sofa cushions
    draw.line([(140, 280), (140, 380)], fill=(0, 0, 0), width=1)

    # coffee table (center)
    draw.rectangle([(200, 340), (310, 380)], outline=(0, 0, 0), width=2)
    draw.line([(210, 380), (210, 400)], fill=(0, 0, 0), width=2)
    draw.line([(300, 380), (300, 400)], fill=(0, 0, 0), width=2)

    # lamp (left side, to be moved)
    # pole
    draw.line([(90, 200), (90, 280)], fill=(0, 0, 0), width=2)
    # shade (triangle)
    draw.polygon([(70, 200), (110, 200), (90, 160)], outline=(0, 0, 0))

    # window on wall
    draw.rectangle([(300, 130), (420, 230)], outline=(0, 0, 0), width=2)
    draw.line([(360, 130), (360, 230)], fill=(0, 0, 0), width=1)
    draw.line([(300, 180), (420, 180)], fill=(0, 0, 0), width=1)

    # --- red arrow annotation (move lamp to right) ---
    # Arrow from lamp position to new position on right
    arrow_color = (255, 0, 0)   # red

    # arrow shaft
    draw.line([(90, 240), (360, 240)], fill=arrow_color, width=3)

    # arrowhead
    draw.polygon(
        [(360, 230), (380, 240), (360, 250)],
        fill=arrow_color
    )

    # circle around lamp (what to move)
    draw.ellipse([(60, 150), (120, 290)], outline=arrow_color, width=2)

    img.save(save_path)
    print(f"[Demo] Demo sketch saved: {save_path}")
    return save_path


def run_pipeline(args):
    """Run the full pipeline with given arguments."""
    import os
    os.chdir(Path(__file__).parent)

    from pipeline import SketchRefinementPipeline, PipelineConfig

    config = PipelineConfig(
        controlnet_type     = args.controlnet,
        num_inference_steps = args.steps,
        guidance_scale      = args.guidance,
        controlnet_scale    = args.cn_scale,
        num_variants        = args.variants,
        seed                = args.seed if args.seed >= 0 else None,
        output_dir          = args.output_dir,
        save_intermediates  = True,
        debug               = args.debug,
    )

    pipeline = SketchRefinementPipeline(config)

    try:
        pipeline.load()

        result = pipeline.run(
            annotated_sketch_path = args.sketch,
            base_sketch_path      = args.base if args.base else None,
            user_text             = args.text,
            prompt_override       = args.prompt if args.prompt else None,
        )

        result.show_summary()

    finally:
        pipeline.unload()


def main():
    parser = argparse.ArgumentParser(
        description="Sketch → Refined 2D Image Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a sketch file
  python run.py --sketch my_sketch.png

  # With user text annotation description
  python run.py --sketch my_sketch.png --text "move the lamp to the right"

  # With both annotated and clean base sketch
  python run.py --sketch annotated.png --base clean.png --text "add pattern to sofa"

  # Override auto-detection with manual prompt
  python run.py --sketch my_sketch.png --prompt "move the chair to the left side"

  # Generate demo sketch and run pipeline (for testing)
  python run.py --demo

  # Higher quality output (slower)
  python run.py --sketch my_sketch.png --steps 30 --variants 4

  # Use softedge ControlNet for cleaner sketches
  python run.py --sketch clean_sketch.png --controlnet softedge
        """
    )

    parser.add_argument(
        "--sketch", type=str, default=None,
        help="Path to annotated sketch image"
    )
    parser.add_argument(
        "--base", type=str, default=None,
        help="Path to clean sketch (without annotations) — improves detection"
    )
    parser.add_argument(
        "--text", type=str, default="",
        help="Optional text description of intended edit"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Manual prompt override (skips annotation detection)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate a demo sketch and run pipeline"
    )
    parser.add_argument(
        "--controlnet", type=str, default="scribble",
        choices=["scribble", "softedge", "lineart", "canny"],
        help="ControlNet type (default: scribble)"
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of denoising steps (default: 20, max recommended: 30)"
    )
    parser.add_argument(
        "--guidance", type=float, default=7.5,
        help="Guidance scale / prompt adherence (default: 7.5)"
    )
    parser.add_argument(
        "--cn-scale", type=float, default=0.9,
        help="ControlNet scale / sketch adherence (default: 0.9)"
    )
    parser.add_argument(
        "--variants", type=int, default=2,
        help="Number of output variants to generate (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (-1 for random, default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output directory (default: outputs/)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    if args.demo:
        args.sketch = generate_demo_sketch("demo_sketch.png")
        if not args.text:
            args.text = "move the lamp to the right side"
        print(f"[Demo] Using demo sketch: {args.sketch}")
        print(f"[Demo] Using text: '{args.text}'")
    elif args.sketch is None:
        parser.print_help()
        print("\nError: provide --sketch or --demo")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
