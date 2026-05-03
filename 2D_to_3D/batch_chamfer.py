"""
batch_chamfer.py - Run Chamfer Distance for every object in input/.

For each <Object> folder in input/, computes:
  - GT vs base sketch mesh   (output/baseSketch/<Object>/.../mesh.glb)
  - GT vs refined image mesh (output/refinedImage/<Object>/.../mesh.glb)

Prints a tabular summary and writes results to chamfer_results.csv.
"""

import csv
import sys
from pathlib import Path

from evaluate_chamfer import compute_chamfer

ROOT = Path(__file__).parent
INPUT_DIR = ROOT / "input"
BASE_OUT = ROOT / "output" / "baseSketch"
REFINED_OUT = ROOT / "output" / "refinedImage"

OBJECTS = [
    "Apple", "Chair", "Donut", "Flower", "FlowerPot",
    "HoneyDipper", "Pizza", "Sunglasses", "Teapot", "TeddyBear",
]

N_POINTS = 10000


def find_gt(obj):
    candidates = list((INPUT_DIR / obj).glob("*.glb"))
    return candidates[0] if candidates else None


def find_mesh(parent, obj):
    """Locate mesh.glb under output/<parent>/<obj>/ (handles flat or nested layouts and case-insensitive folder)."""
    obj_dir = parent / obj
    if not obj_dir.exists():
        for sibling in parent.iterdir():
            if sibling.is_dir() and sibling.name.lower() == obj.lower():
                obj_dir = sibling
                break
        else:
            return None
    matches = list(obj_dir.rglob("mesh.glb"))
    return matches[0] if matches else None


def main():
    rows = []
    print(f"{'Object':<14} {'GT vs Base':>14} {'GT vs Refined':>16}")
    print("-" * 46)

    for obj in OBJECTS:
        gt = find_gt(obj)
        base = find_mesh(BASE_OUT, obj)
        refined = find_mesh(REFINED_OUT, obj)

        cd_base = cd_refined = None
        if gt and base:
            try:
                cd_base = compute_chamfer(str(gt), str(base), n_points=N_POINTS)
            except Exception as e:
                print(f"  [WARN] {obj} base failed: {e}", file=sys.stderr)
        if gt and refined:
            try:
                cd_refined = compute_chamfer(str(gt), str(refined), n_points=N_POINTS)
            except Exception as e:
                print(f"  [WARN] {obj} refined failed: {e}", file=sys.stderr)

        base_str = f"{cd_base:.6f}" if cd_base is not None else "N/A"
        refined_str = f"{cd_refined:.6f}" if cd_refined is not None else "N/A"
        print(f"{obj:<14} {base_str:>14} {refined_str:>16}")

        rows.append({
            "object": obj,
            "gt_path": str(gt) if gt else "",
            "base_path": str(base) if base else "",
            "refined_path": str(refined) if refined else "",
            "chamfer_gt_vs_base": base_str,
            "chamfer_gt_vs_refined": refined_str,
        })

    csv_path = ROOT / "chamfer_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
