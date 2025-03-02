"""
Stable Diffusion Textual Inversion Module

This module provides functionality for teaching Stable Diffusion models new concepts
via textual inversion. Users can train custom tokens to represent new objects or styles
with just a few example images.
"""

from .dataset import TextualInversionDataset
from .utils import image_grid, freeze_params, save_progress
from .train import train_textual_inversion
from .inference import run_inference

__version__ = "0.1.0"