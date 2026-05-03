# Sketch to 3D

Sketch to 3D sketch-guided image refinement and reconstruction pipeline that utilizes Stable Diffusion XL and ControlNet. The application includes a Streamlit-based interactive web interface and an OpenCV-based annotation detector that classifies colored annotations into structural adjustments and prompt modifications.

## Annotation Processing Guide

The application's annotation processing module isolates specific colors from your sketch using HSV thresholding and reads the text embedded within those colors:

* **Red text**: Maps to the name or identity of the object (e.g., assigned as `st.session_state.obj_name` in the Streamlit app).
* **Black text**: Defines the color, texture, or style parameter of the object (e.g., used to form `"with a {annotations.get('Black', 'standard')}..."` in the prompt generation logic).
* **Blue text**: Extracted as additional features to modify the refined image generation.


---

## Commands to Run

### 1. Install Dependencies
Ensure you have the required packages installed in your environment:
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA
Depending on your CUDA version, install PyTorch with the appropriate drivers:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Start the Web Application
To run the interactive Streamlit user interface:
```bash
streamlit run main.py
```

### 4. Running the CLI Tool (Optional)
You can run the pipeline directly using the CLI entry point with a base sketch:
```bash
python run.py --sketch annotated.png --base clean.png --text "move the object to the right"
```

---

## Model Architecture and Pipeline

The application loads the Stable Diffusion XL pipeline along with the necessary MistoLine and VAE components:

```python
print("Initializing SDXL and Anyline Models...")
anyline = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
controlnet = ControlNetModel.from_pretrained("TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
```

---

## 2D → 3D Reconstruction

After refining your 2D asset, you can convert the output image into a 3D mesh using our enhanced TripoSR implementation. This branch includes several modifications to the base [TripoSR repository](https://github.com/VAST-AI-Research/TripoSR):

## 1. Commands to Run

Ensure you have the required packages installed in your environment:
```bash
pip install -r requirements.txt
pip install --upgrade setuptools
```
* **Named Output Directories**: Results are saved in folders named after the input image (e.g., `output/chair/`) for easier organization.
* **Dual Mesh Export**: Automatically exports generated meshes in both `.obj` and `.glb` formats.
* **Pre-trained Model Integration**:
```python
from tsr.system import TSR
model = TSR.from_pretrained('stabilityai/TripoSR')
```
### 2. Running 3D Reconstruction
To convert your refined image into a 3D mesh:
```bash
python run.py path/to/your/image.png --output-dir output/
```

### 3. Start the 3D User Interface
You can run Gradio app for 3D reconstruction:
```bash
# Original TripoSR UI
python gradio_app.py
```
---
## Evaluation Metrics

This repository includes chamfer distance metric to evaluate the quality of the 3D reconstruction against ground truth data.

### Individual Metric Scripts
* **Chamfer Distance (Geometric Similarity)**:
  ```bash
  python evaluate_chamfer.py --input path/to/gt.glb --output path/to/pred.glb

### Batch Processing
To automate evaluation across the entire `input/` dataset:
* **Batch Chamfer**: `python batch_chamfer.py`
Results are logged to `chamfer_results.csv`.
---
## Citation

If you use this work, please cite the original TripoSR paper:

```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}
```

