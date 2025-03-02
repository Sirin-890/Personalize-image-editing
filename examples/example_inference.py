"""
Example script for running inference with a trained textual inversion model.
"""

import os
import argparse
import torch
import yaml
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from src.utils import image_grid


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained textual inversion model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="sd-concept-output",
                        help="Path to trained model directory")
    parser.add_argument("--prompt", type=str, 
                        help="Prompt to use for generation (must include placeholder token)")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to generate")
    parser.add_argument("--num_rows", type=int, default=1,
                        help="Number of rows in output grid")
    parser.add_argument("--output_file", type=str, default="output.png",
                        help="File to save generated images")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if args.prompt:
        config["inference"]["prompt"] = args.prompt
    
    prompt = config["inference"]["prompt"]
    num_samples = args.num_samples
    num_rows = args.num_rows
    output_file = args.output_file
    
    print(f"Loading model from {args.model_path}")
    
    # Set up the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            args.model_path, subfolder="scheduler"
        ),
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Generating images with prompt: '{prompt}'")
    
    # Generate images
    all_images = []
    for _ in range(num_rows):
        images = pipe(
            [prompt] * num_samples, 
            num_inference_steps=config["inference"]["num_inference_steps"],
            guidance_scale=config["inference"]["guidance_scale"]
        ).images
        all_images.extend(images)
    
    # Create and save grid
    print(f"Creating image grid with {len(all_images)} images")
    grid = image_grid(all_images, num_rows, num_samples)
    
    # Save the result
    print(f"Saving result to {output_file}")
    grid.save(output_file)
    print("Done!")


if __name__ == "__main__":
    main()