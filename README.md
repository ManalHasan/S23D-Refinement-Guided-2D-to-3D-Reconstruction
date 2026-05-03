# Sketch to #D

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

After refining your 2D asset using Sketch2Blend, you can convert the output image into a 3D mesh using the TripoSR repository ([https://github.com/VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR))
model = TSR.from_pretrained('stabilityai/TripoSR')
```
