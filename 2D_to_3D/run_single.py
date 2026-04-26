#!/usr/bin/env python3
"""
run_single.py — Single image → TripoSR → 3D (.obj + .glb) + preview renders.

Usage:
    python run_single.py <image_path> [--output_dir outputs] [--no-rembg] [--resolution 256] [--device cpu]

Example:
    python run_single.py inputs/refined_images/chair.png --output_dir outputs
"""

import argparse
import torch
from triposr_utils import full_pipeline, load_model, get_device


def main():
    parser = argparse.ArgumentParser(
        description="Run TripoSR 3D reconstruction on a single image"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output_dir", default="outputs",
                        help="Base output directory (default: outputs)")
    parser.add_argument("--no-rembg", action="store_true",
                        help="Skip background removal")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Marching cubes resolution (default: 256)")
    parser.add_argument("--name", default=None,
                        help="Output name (default: input filename stem)")
    parser.add_argument("--device", default=None,
                        help="Device to use (cpu, cuda, mps). Default: stable auto-detect.")
    args = parser.parse_args()

    # Determine device
    force_cpu = (args.device == "cpu")
    device = get_device(force_cpu=force_cpu)
    if args.device and args.device != "cpu":
        device = torch.device(args.device)

    model, device = load_model(device=device)

    result = full_pipeline(
        image_path=args.image,
        output_dir=args.output_dir,
        model=model,
        device=device,
        name=args.name,
        remove_bg=not args.no_rembg,
        mc_resolution=args.resolution,
    )

    print(f"\n{'='*50}")
    print("Results:")
    print(f"  OBJ:      {result['obj']}")
    print(f"  GLB:      {result['glb']}")
    print(f"  Grid:     {result['grid']}")
    print(f"  Previews: {len(result['previews'])} images")
    print(f"\nView your .glb at: https://gltf-viewer.donmccurdy.com")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
