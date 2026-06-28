import torch
from diffusers import AutoPipelineForText2Image
import os
from tqdm import tqdm
import argparse

# Supported model names
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}

INVERTED_MODELS = {v: k for k, v in MODELS.items()}

# Save latents to disk
def save_latents(latents, latent_path):
    print(f"Saving latents to {latent_path}")
    os.makedirs(os.path.dirname(latent_path), exist_ok=True)
    torch.save(latents, latent_path)

# Save image to disk
def save_image(image, image_path):
    print(f"Saving image to {image_path}")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path, format="PNG")

# Main generator class for images and latents
class ImageLatentGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", dtype=torch.float16,
                 ddpm=False):
        # Initialize pipeline and settings
        self.device = device
        self.dtype = dtype
        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_name)
        self.pipeline = self.pipeline.to(device="cuda", dtype=self.dtype)
        self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)
        self.temp_latents = []
        self.model_name = model_name
        self.ddpm = ddpm

    # Hook to capture latents during forward pass
    def hook(self, module, input, output):
        self.temp_latents.append(input[0].detach().cpu())

    # Generate images and latents for given prompts
    def generate_images_and_latents(self, prompts, filenames, seed=31, skip_generated=False, num_images=1):
        generator = torch.Generator(self.device).manual_seed(seed)
        # Handle multiple images per prompt
        if num_images > 1:
            new_prompts, new_filenames = [], []
            for i, prompt in enumerate(prompts):
                original_filename = filenames[i]
                for j in range(num_images):
                    new_prompts.append(prompt)
                    new_filenames.append(f"{original_filename}_{j}")
            prompts, filenames = new_prompts, new_filenames

        # Main generation loop
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            short_name = INVERTED_MODELS.get(self.model_name, "unknown_model")
            if self.ddpm:
                short_name += "_ddpm"

            file_name_temp = filenames[idx].split("/")[-1]
            file_name_base = file_name_temp.split(".")[0]
            image_version_index = int(file_name_base.split("_")[-1])
            image_path = f"{output_path}/images/{short_name}/{file_name_base}.png"
            latent_path = f"{output_path}/latents/{short_name}/{file_name_base}.pt"

            # Skip if image already exists
            if skip_generated and os.path.exists(image_path):
                print(f"Skipping already generated image: {image_path}")
                continue

            # Register hook to capture latents
            unet = self.pipeline.unet
            hook_handle = unet.register_forward_hook(self.hook)

            # Generate image
            image = self.pipeline(prompt, generator=generator, eta=1 if self.ddpm else 0).images[0]

            hook_handle.remove()
            save_image(image, image_path=image_path)
            save_latents(self.temp_latents, latent_path=latent_path)
            self.temp_latents = []

if __name__ == '__main__':

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure a GPU is properly configured.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate images and latents from captions.")
    parser.add_argument("--caption_path", type=str, help="Path to the caption file.")
    parser.add_argument("--output_path", type=str, help="Path to save the generated images and latents.")
    parser.add_argument("--model_name", type=str, help="Name of the model to use.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to use.")
    parser.add_argument("--ddpm", action="store_true", help="Use DDPM model.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate per prompt.")

    args = parser.parse_args()

    caption_path = args.caption_path
    model_name = args.model_name
    ddpm = args.ddpm
    output_path = args.output_path
    dataset_name = args.dataset_name
    num_images = args.num_images

    # Validate input paths and model names
    if caption_path is not None and not os.path.exists(caption_path):
        raise FileNotFoundError(f"Caption path '{caption_path}' does not exist.")

    if model_name not in MODELS:
        raise ValueError(f"Model name '{model_name}' is not recognized. Valid options are: {list(MODELS.keys())}.")

    # Load dataset and captions
    if dataset_name == "cc12m":
        from datasets import load_from_disk
        dataset = load_from_disk(caption_path)
        captions = dataset["caption_llava_short"]
        gt_path_current_index = dataset["key"]
    elif dataset_name == "genai-bench":
        from datasets import load_dataset
        dataset = load_dataset("BaiqiL/GenAI-Bench")
        captions = dataset["train"]["Prompt"]
        gt_path_current_index = dataset["train"]["Index"]
    else:
        raise ValueError(
            f"Dataset name '{dataset_name}' is not recognized. Valid options are: ['cc12m', 'genai-bench'].")

    print("Amount of captions to generate:", len(captions))
    generator = ImageLatentGenerator(model_name=MODELS[model_name], ddpm=ddpm)
    generator.generate_images_and_latents(captions, gt_path_current_index, skip_generated=True, num_images=num_images)