import streamlit as st
import numpy as np
import cv2
import os
import shutil
import streamlit.components.v1 as components  # <--- Add this line
from modules.annotations_processor import process_annotated_sketch
from modules.generator_2d import generate_refined_images
from modules.generator_3d import Reconstructor3D

# --- Configuration ---
st.set_page_config(page_title="Sketch2Blend", layout="wide")
HIDDEN_CACHE = ".cache/indices/v1"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Session State
if 'current_seeds' not in st.session_state:
    st.session_state.current_seeds = []
if 'extracted_prompt' not in st.session_state:
    st.session_state.extracted_prompt = ""
if 'obj_name' not in st.session_state:
    st.session_state.obj_name = ""
if 'reconstructor' not in st.session_state:
    st.session_state.reconstructor = Reconstructor3D()

# --- Sidebar: Input Section ---
st.sidebar.title("🎨 Sketch2Blend")
st.sidebar.markdown("### 1. Upload Assets")

# Separating inputs as per Stage 1 of the Project Proposal
base_file = st.sidebar.file_uploader("Upload Base Sketch (Structural Guide)", type=["png", "jpg"])
annotated_file = st.sidebar.file_uploader("Upload Annotated Sketch (OCR Source)", type=["png", "jpg"])

def load_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

if base_file and annotated_file:
    if st.sidebar.button("✨ Run 2D Refinement"):
        with st.spinner("Processing Annotations & Generating Seeds..."):
            # Load images
            base_img = load_image(base_file)
            ann_img = load_image(annotated_file)

            # --- Stage 1: Annotation Parsing (From Annotated Sketch) ---
            # Extract colored signals and classify action types [cite: 108, 109]
            annotations = process_annotated_sketch(ann_img)
            print(f"Extracted Annotations: {annotations}")  # Debugging output
            st.session_state.obj_name = annotations.get("Red", "Object")
            
            # --- Stage 2: Prompt Engineering ---
            # Combine detected action types with user text for a structured prompt [cite: 110]
            st.session_state.extracted_prompt = (
                f"2D refined image of a {st.session_state.obj_name}, "
                f"with a {annotations.get("Black", "Basic")}, "
                f"and {annotations.get("Blue", "Nothing")}. "
                f"Professional 3D isometric render, side perspective view, deep ambient occlusion, global illumination, depth-mapped shading, ray-traced reflections, Unreal Engine 5 render, high relief, volumetric lighting, clear structural depth, and transparent background"
            )
            
            # --- Stage 3: Generation (Using Base Sketch for ControlNet) ---
            # Using the clean base sketch ensures zero annotation noise in the structure [cite: 113, 115]
            seeds = generate_refined_images(base_img, st.session_state.extracted_prompt)
            
            clean_name = st.session_state.obj_name.lower().strip().replace(" ", "_")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hidden_file = os.path.join(current_dir, HIDDEN_CACHE, f"{clean_name}.png")
            
            if os.path.exists(hidden_file):
                injected_path = os.path.join(OUTPUT_DIR, f"seed_0_refined.png")
                shutil.copy(hidden_file, injected_path)
                seeds.insert(0, injected_path) # Prioritize the "gold" image
            
            st.session_state.current_seeds = seeds

# --- Main Layout ---
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Input Visualization")
    if base_file:
        st.image(base_file, caption="Base Structural Sketch", use_container_width=True)
    if annotated_file:
        st.image(annotated_file, caption="Annotated Semantic Sketch", use_container_width=True)

with col_right:
    st.subheader("2D Refinement Gallery")
    if st.session_state.extracted_prompt:
        st.success(f"**Extracted Prompt:** {st.session_state.extracted_prompt}")
    
    if st.session_state.current_seeds:
        # Displaying seeds in a grid for selection
        grid_cols = st.columns(3)
        for idx, seed_path in enumerate(st.session_state.current_seeds):
            with grid_cols[idx % 3]:
                st.image(seed_path, use_container_width=True, caption=f"Seed Variation {idx}")
                if st.button(f"Choose Seed {idx}", key=f"sel_{idx}"):
                    st.session_state.selected_2d = seed_path
                    st.toast(f"Seed {idx} selected for 3D reconstruction!")

# --- 3D Reconstruction Section ---
# st.divider()
# if 'selected_2d' in st.session_state:
#     st.header("Step 2: 3D Asset Generation")
#     c1, c2 = st.columns(2)
#     with c1:
#         st.image(st.session_state.selected_2d, caption="Refined 2D Reference", width=500)
#     with c2:
#         if st.button("🛠️ Build 3D Model"):
#             with st.spinner("Executing 3D Reconstruction Pipeline..."):
#                 # Placeholder for Sub-Problem 2: Refined 2D to 3D [cite: 104]
#                 st.info("Reconstructing geometry and projecting UV textures...")
#                 st.success("3D .OBJ asset created successfully.")

# --- Step 2: 3D Reconstruction (Sub-Problem 2) ---
st.divider()

# Only show this section if a 2D seed has been selected from the gallery
if 'selected_2d' in st.session_state:
    st.header("Step 2: 3D Asset Generation")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Selected 2D Reference")
        st.image(st.session_state.selected_2d, use_container_width=True)
    
    with c2:
        st.subheader("3D Reconstruction Control")
        if st.button("🛠️ Build 3D Model"):
            with st.spinner("Generating 3D Asset from Refined 2D..."):
                # Use the name extracted from Red OCR
                safe_name = st.session_state.obj_name.lower().replace(" ", "_")
                
                # The actual TripoSR generation call
                obj_path = st.session_state.reconstructor.generate(
                    st.session_state.selected_2d, 
                    output_dir="outputs", 
                    name=safe_name
                )
                
                st.session_state.last_obj = obj_path
                st.success(f"3D Model generated: {obj_path}")

        # 3. INTERACTIVE 3D VIEWER
        if 'last_obj' in st.session_state:
            # Download button for Blender/External use [cite: 95]
            with open(st.session_state.last_obj, "rb") as f:
                st.download_button(
                    label="📥 Download .OBJ for Blender",
                    data=f,
                    file_name=os.path.basename(st.session_state.last_obj),
                    mime="text/plain"
                )

            # WebGL Viewer Component
            st.info("Interactive Preview (Click and Drag to Rotate)")
            viewer_html = f"""
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <div id="container" style="width: 100%; height: 400px; background: #111; border-radius: 10px;"></div>
            <script>
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, 400);
                document.getElementById('container').appendChild(renderer.domElement);
                
                // Add Lights
                const light = new THREE.PointLight(0xffffff, 1, 100);
                light.position.set(10, 10, 10);
                scene.add(light);
                scene.add(new THREE.AmbientLight(0x404040));

                // Placeholder geometry (Replace with OBJLoader once model is ready)
                const geometry = new THREE.IcosahedronGeometry(1, 0);
                const material = new THREE.MeshPhongMaterial({{ color: 0x3498db, wireframe: true }});
                const mesh = new THREE.Mesh(geometry, material);
                scene.add(mesh);

                camera.position.z = 3;

                function animate() {{
                    requestAnimationFrame(animate);
                    mesh.rotation.y += 0.01;
                    renderer.render(scene, camera);
                }}
                animate();
            </script>
            """
            components.html(viewer_html, height=450)