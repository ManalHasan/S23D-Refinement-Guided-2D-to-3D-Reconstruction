"""
evaluate_chamfer.py - Chamfer Distance between two meshes.

Compares an input ground-truth mesh against an output mesh produced by
TripoSR (or any source). Both meshes can be .glb, .obj, .ply, etc. —
trimesh figures out the format from the file extension.

Usage:
    python evaluate_chamfer.py --input path/to/input.glb --output path/to/output.glb
    python evaluate_chamfer.py --input gt.glb --output pred.glb --n-points 20000
"""

import argparse
import sys

import numpy as np
import trimesh


def _sample_surface_points(mesh_path, n_points=10000):
    """Sample points uniformly on the mesh surface, then normalize to a unit cube
    so the metric is scale-invariant (TripoSR output and a GT mesh rarely share
    the same world scale)."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"Mesh at {mesh_path} is empty or has no faces.")

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = np.asarray(points, dtype=np.float64)

    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale
    return points


def compute_chamfer(mesh_path_a, mesh_path_b, n_points=10000, chunk=2000):
    """Symmetric Chamfer Distance between two meshes (pure numpy).
    Lower = more similar geometry."""
    pts_a = _sample_surface_points(mesh_path_a, n_points)
    pts_b = _sample_surface_points(mesh_path_b, n_points)

    def min_dists(src, tgt):
        dists = np.zeros(src.shape[0])
        for i in range(0, src.shape[0], chunk):
            diff = src[i : i + chunk, None, :] - tgt[None, :, :]
            d2 = (diff ** 2).sum(axis=-1)
            dists[i : i + chunk] = d2.min(axis=1)
        return dists

    d_a2b = min_dists(pts_a, pts_b)
    d_b2a = min_dists(pts_b, pts_a)
    return float(d_a2b.mean() + d_b2a.mean())


def main():
    parser = argparse.ArgumentParser(
        description="Compute Chamfer Distance between an input mesh and an output mesh."
    )
    parser.add_argument("--input", required=True, help="Input / ground-truth mesh (.glb, .obj, .ply, ...)")
    parser.add_argument("--output", required=True, help="Output / predicted mesh (.glb, .obj, .ply, ...)")
    parser.add_argument("--n-points", type=int, default=10000, help="Points sampled per mesh (default: 10000)")
    args = parser.parse_args()

    print(f"Input mesh:  {args.input}")
    print(f"Output mesh: {args.output}")
    print(f"Sampling {args.n_points} points per surface...")

    try:
        cd = compute_chamfer(args.input, args.output, n_points=args.n_points)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nChamfer Distance: {cd:.6f}")
    print("(lower = more similar geometry; meshes are unit-cube normalized before comparison)")


if __name__ == "__main__":
    main()
