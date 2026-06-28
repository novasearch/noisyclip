import torch
from PIL import Image
import os
from tqdm import tqdm
import argparse
from datasets import load_from_disk

# Converts latent tensors to RGB images using fixed weights and biases
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35)
    )
    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)  # Rearrange dimensions for image
    return Image.fromarray(image_array)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert latents to images")
    parser.add_argument("--latents_path", type=str, help="Path to the directory containing latent files", required=True)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Dataset name to filter the latents, if applicable. Defaults to None.")
    args = parser.parse_args()
    latents_path = args.latents_path
    if not latents_path.endswith('/'):
        latents_path += '/'
    dataset_path = args.dataset_path


    if not os.path.exists(latents_path):
        raise ValueError(f"The provided path {latents_path} does not exist.")

    # Load dataset keys for filtering if provided
    if dataset_path:
        ds = load_from_disk(dataset_path)
        ds_keys = set(ds["key"])

    # Iterate over latent files and convert to images
    for file in tqdm(os.listdir(latents_path), desc="Converting latents to images"):
        if not file.endswith(".pt"):
            continue
        if dataset_path:
            file_key = file.split("_")[0]
            if file_key not in ds_keys:
                continue
        latents = torch.load(latents_path + file)
        if isinstance(latents, list):
            latents = [latent.to("cuda") for latent in latents]
        else:
            latents = latents.to("cuda")
        # Create output directory for images
        output_dir = f"{latents_path}/{file.split('.')[0]}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        files = os.listdir(output_dir)
        # Save each latent as an image
        for i in range(len(latents)):
            if files and f"{i}.png" in files:
                continue
            img = latents_to_rgb(latents[i])
            img.save(f"{output_dir}/{i}.png")
        print(f"Converted {file} to images in {output_dir}")