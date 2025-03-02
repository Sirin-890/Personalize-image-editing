import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

class InferenceModel:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
            
        # Load the placeholder token
        if os.path.exists(os.path.join(self.model_path, "token_identifier.txt")):
            with open(os.path.join(self.model_path, "token_identifier.txt"), "r") as f:
                self.placeholder_token = f.read().strip()
        else:
            self.placeholder_token = None
            
        # Load model with custom scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(self.model_path, subfolder="scheduler"),
            torch_dtype=torch.float16,
        ).to(self.device)
        
    def generate(self, prompt, num_samples=1, num_inference_steps=30, guidance_scale=7.5):
        """
        Generate images from a prompt
        
        Args:
            prompt (str): Text prompt for image generation
            num_samples (int): Number of images to generate
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for classifier-free guidance
            
        Returns:
            list: List of PIL.Image objects
        """
        # If placeholder token is known, make sure it's in the prompt
        if self.placeholder_token and self.placeholder_token not in prompt:
            print(f"Warning: Your prompt doesn't contain the placeholder token: {self.placeholder_token}")
            
        images = self.pipe(
            [prompt] * num_samples, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        ).images
        
        return images
        
    def create_image_grid(self, images, rows=1):
        """Create a grid of images"""
        from .utils import image_grid
        cols = len(images) // rows
        return image_grid(images, rows, cols)