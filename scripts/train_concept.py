#!/usr/bin/env python
import argparse
import os
import logging
import yaml
from src.train import train_textual_inversion
from src.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train a textual inversion concept")
    parser.add_argument(
        "--concept_name", type=str, required=True, help="Name of the concept to train"
    )
    parser.add_argument(
        "--concept_type", type=str, default="object", choices=["object", "style"],
        help="Type of concept to train, 'object' or 'style'"
    )
    parser.add_argument(
        "--placeholder_token", type=str, required=True,
        help="Token to use as a placeholder for the concept, e.g. '<my-token>'"
    )
    parser.add_argument(
        "--initializer_token", type=str, required=True,
        help="Token to use as initializer, should be a single token, e.g. 'toy'"
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the directory containing training images"
    )
    parser.add_argument(
        "--pretrained_model", type=str, default="stabilityai/stable-diffusion-2",
        help="Pretrained model path or identifier from Hugging Face"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory to save the model and embeddings"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=2000,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=250,
        help="Number of steps between saving checkpoints"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Resolution for training images"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for training"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file, will override other arguments if provided"
    )
    
    return parser.parse_args()

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    # If config file is provided, load it
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {args.config}")
    
    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"output/{args.concept_name}"
    
    # Combine config and args, prioritizing config
    train_kwargs = {
        "pretrained_model_name_or_path": config.get("pretrained_model", args.pretrained_model),
        "train_data_dir": config.get("image_path", args.image_path),
        "output_dir": config.get("output_dir", args.output_dir),
        "placeholder_token": config.get("placeholder_token", args.placeholder_token),
        "initializer_token": config.get("initializer_token", args.initializer_token),
        "learnable_property": config.get("concept_type", args.concept_type),
        "resolution": config.get("resolution", args.resolution),
        "train_batch_size": config.get("train_batch_size", args.train_batch_size),
        "learning_rate": config.get("learning_rate", args.learning_rate),
        "max_train_steps": config.get("max_train_steps", args.max_train_steps),
        "save_steps": config.get("save_steps", args.save_steps),
        "seed": config.get("seed", args.seed),
    }
    
    # Log training parameters
    logger.info(f"Training concept: {args.concept_name}")
    logger.info(f"Type: {train_kwargs['learnable_property']}")
    logger.info(f"Placeholder token: {train_kwargs['placeholder_token']}")
    logger.info(f"Initializer token: {train_kwargs['initializer_token']}")
    logger.info(f"Image path: {train_kwargs['train_data_dir']}")
    logger.info(f"Output directory: {train_kwargs['output_dir']}")
    
    # Train the model
    train_textual_inversion(**train_kwargs)
    
    logger.info(f"Training complete! Model saved to {train_kwargs['output_dir']}")
    
if __name__ == "__main__":
    main()