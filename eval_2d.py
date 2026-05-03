import os
import torch
import clip
from PIL import Image
from cleanfid import fid

# 1. Configuration Constants
DEVICE_ACCELERATION = "cuda" if torch.cuda.is_available() else "cpu"

# Construct descriptive paths
OUTPUT_DIRECTORY = #add your generated images directory here, e.g. "outputs"
ground_truth_directory = #add your ground truth path here, e.g. "ground_truth/standard_objects"
sample_generated_image = #add your generated image path here, e.g. "outputs/seed_0_refined.png"

# 2. Load CLIP Model
print("Loading CLIP model...")
clip_model, preprocess_image = clip.load("ViT-B/32", device=DEVICE_ACCELERATION)

def calculate_clip_score(image_path, text_prompt):
    image = preprocess_image(Image.open(image_path)).unsqueeze(0).to(DEVICE_ACCELERATION)
    text_tokens = clip.tokenize([text_prompt]).to(DEVICE_ACCELERATION)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)

        # Cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()

    return similarity

def calculate_fid_score(generated_dir, reference_dir):
    print(f"Calculating FID between {generated_dir} and {reference_dir}...")
    score = fid.compute_fid(
        generated_dir, 
        reference_dir,
        num_workers=4,
        batch_size=16,
        device=DEVICE_ACCELERATION
    )
    return score

if __name__ == "__main__":
    # Define descriptive prompt used for the test
    prompt_description = #add the prompt that was exracted and used for generation, e.g. "2D refined image of a chair, with a wooden texture, and ergonomic features. Realistic 3D render with natural soft lighting, diffuse shading, gentle ambient occlusion, physically plausible materials, subsurface scattering, soft shadows, high color fidelity, organic texture, no reflections, no metallic appearance, no plastic look, transparent background"

    
    if os.path.exists(sample_generated_image):
        # Calculate CLIP
        clip_score = calculate_clip_score(sample_generated_image, prompt_description)
        
        # Calculate FID
        fid_score = calculate_fid_score(OUTPUT_DIRECTORY, ground_truth_directory)
        
        print("\n==============================")
        print(" EVALUATION METRICS SUMMARY")
        print("==============================")
        print(f"CLIP Score (Semantic Alignment): {clip_score:.4f}")
        print(f"FID Score (Generative Quality) : {fid_score:.4f}")
        print("==============================\n")
        
    else:
        print(f"Generated image missing at: {sample_generated_image}")
        print("Please run your 2D generation pipeline first to produce seeds.")