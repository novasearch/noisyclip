
<div style="text-align: center;">
    <img src="/static/images/noisyclip_logo.png" alt="NoisyCLIP" width="250" height="250">
</div>

# Early Estimation of Language to Latent Alignment in Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2512.08505-b31b1b.svg)](https://arxiv.org/abs/2512.08505)
[![Project](https://img.shields.io/badge/Project-Website-9cf.svg)](https://novasearch.github.io/noisyclip/)



This is the official repository for Early Estimation of Language to Latent Alignment in Diffusion Models (ECCV 2026). We propose **NoisyCLIP**, a noise-aware twin-tower model that enables early language-to-latent alignment estimation, transforming alignment assessment from an expensive final check into a continuous monitoring tool.

## Code Structure

`src/corrupt_prompts.py` - generates non-factual (corrupted) text prompts from original captions for robust model testing

`src/generate_latents.py` - uses Stable Diffusion XL to generate images and latent representations from text prompts

`src/latent_to_image.py` - converts saved latent tensors into RGB representations

`src/train_model_classic.py` - trains a CLIP or SigLIP model using the preprocessed datasets

`appendix.pdf` - supplementary material with additional results and ablations

## Usage

1. **Generate Corrupted Prompts** — prepare a dataset and run `corrupt_prompts.py` to create non-factual captions.

2. **Generate Images and Latents** — use `generate_latents.py` to create latents and images from your text prompts.

3. **Convert Latents to Images** — run `latent_to_image.py` to create RGB representations from the latent tensors.

4. **Train the Model** — use `train_model_classic.py` to train your model. See the script arguments for customization.

## Citation
If you find NoisyCLIP useful for your research and applications, please cite using this BibTeX:
```
TO BE UPDATED
```
