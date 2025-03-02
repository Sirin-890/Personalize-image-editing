#!/usr/bin/env python
import argparse
import os
from src.utils import download_images, setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Download images for training")
    parser.add_argument(
        "--urls", type=str, nargs="+", required=True,
        help="List of URLs to download images from"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save downloaded images"
    )
    
    return parser.parse_args()

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download images
    logger.info(f"Downloading {len(args.urls)} images to {args.output_dir}")
    num_downloaded = download_images(args.urls, args.output_dir)
    
    logger.info(f"Downloaded {num_downloaded} images")
    
if __name__ == "__main__":
    main()