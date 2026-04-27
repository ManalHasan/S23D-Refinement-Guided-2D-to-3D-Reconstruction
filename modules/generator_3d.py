import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# 1. PATH SETUP
# Ensures the TripoSR library is accessible
TRIPOSR_DIR = '/home/mh08438/S23D-Refinement-Guided-2D-to-3D-Reconstruction/TripoSR'
if os.path.exists(TRIPOSR_DIR):
    sys.path.insert(0, TRIPOSR_DIR)

class Reconstructor3D:
    def __init__(self, device=None):
        from tsr.system import TSR
        
        # Select device (CUDA > CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[3D] Loading TripoSR onto {self.device}...")
        
        # Load the stabilityai model
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(8192)
        self.model.to(self.device)
        self.rembg_session = None
        print("[3D] Model loaded successfully.")

    def _preprocess(self, image_path):
        """Matches Stage 3 requirements: Prepare refined 2D for 3D."""
        from tsr.utils import remove_background, resize_foreground
        import rembg
        
        image = Image.open(image_path).convert("RGBA")
        
        if self.rembg_session is None:
            self.rembg_session = rembg.new_session()
        
        # Clean background and resize to expected foreground ratio
        image = remove_background(image, self.rembg_session)
        image = resize_foreground(image, 0.85)
        
        # Composite onto gray background (TripoSR requirement)
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((arr * 255.0).astype(np.uint8))

    def generate(self, image_path, output_dir="outputs", name="model"):
        """
        Executes Sub-Problem 2: Refined 2D to 3D Reconstruction.
        Returns the path to the generated .obj file.
        """
        from tsr.bake_texture import bake_texture
        from tsr.utils import to_gradio_3d_orientation
        import xatlas
        
        # 1. Preprocess the refined 2D image
        processed_img = self._preprocess(image_path)
        
        # 2. Run Inference
        with torch.no_grad():
            scene_codes = self.model([processed_img], device=self.device)

        # 3. Extract Mesh (requires CPU for marching cubes)
        self.model.to("cpu")
        scene_codes = scene_codes.to("cpu")
        
        meshes = self.model.extract_mesh(scene_codes, has_vertex_color=False, resolution=256)
        mesh = to_gradio_3d_orientation(meshes[0])
        
        # 4. Bake Texture
        bake_output = bake_texture(mesh, self.model, scene_codes[0], 2048)
        
        # 5. Export to outputs/
        os.makedirs(output_dir, exist_ok=True)
        obj_path = os.path.join(output_dir, f"{name}.obj")
        tex_path = os.path.join(output_dir, f"{name}_tex.png")
        
        # Save Texture Map
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(tex_path)
        
        # Save OBJ with UVs
        xatlas.export(
            str(obj_path), 
            mesh.vertices[bake_output["vmapping"]], 
            bake_output["indices"], 
            bake_output["uvs"], 
            mesh.vertex_normals[bake_output["vmapping"]]
        )
        
        # Reset model to original device
        self.model.to(self.device)
        
        return obj_path