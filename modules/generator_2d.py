import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from controlnet_aux import AnylineDetector

# Load models once outside the function to avoid re-loading on every 'Reload' click
print("Initializing SDXL and Anyline Models...")
anyline = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
controlnet = ControlNetModel.from_pretrained("TheMistoAI/MistoLine", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("Lykon/dreamshaper-xl-1-0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
pipe.enable_model_cpu_offload()

def generate_refined_images(sketch_img, prompt, seeds=[777, 42, 123]):
    IMAGE_SIZE = (768, 1024)
    NEGATIVE_PROMPT = "background, scenery, landscape, low quality, blurry, distorted, sketch"
    
    # Preprocess
    sketch_pil = Image.fromarray(sketch_img).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    control_image = anyline(sketch_pil, detect_resolution=1280, output_type="pil").resize(IMAGE_SIZE, Image.LANCZOS)
    
    output_paths = []
    for i, seed in enumerate(seeds):
        generator = torch.Generator().manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=control_image,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]

        # White background cleanup logic from your code
        result_np = np.array(result)
        mask = (result_np[:,:,0] > 220) & (result_np[:,:,1] > 220) & (result_np[:,:,2] > 220)
        result_np[mask] = [255, 255, 255]
        
        path = f"outputs/gen_{seed}.png"
        Image.fromarray(result_np).save(path)
        output_paths.append(path)
        
    return output_paths