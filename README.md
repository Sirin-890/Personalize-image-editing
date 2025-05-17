# Stable Diffusion Textual Inversion

## Overview

This project provides a comprehensive toolkit for training and using Textual Inversion with Stable Diffusion models. Textual Inversion is a technique that allows you to teach a Stable Diffusion model new concepts using just a few example images.




## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended
- PyTorch
- Hugging Face Diffusers library

### Clone the Repository

```bash
git clone https://github.com/Sirin-890/stable-diffusion-textual-inversion.git
cd stable-diffusion-textual-inversion
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a New Concept

1. Prepare your training images in a directory
2. Configure your training in `config/default_config.yaml`
3. Run the training script:

```bash
python scripts/train_concept.py \
    --images_path /path/to/your/images \
    --placeholder_token "<my-concept>" \
    --initializer_token "object"
```

### Running Inference

```bash
python examples/example_inference.py \
    --prompt "a <my-concept> in a magical forest" \
    --num_samples 4
```

## Configuration

The project uses a YAML-based configuration system. Key configuration options include:

- Model selection
- Training hyperparameters
- Data augmentation settings
- Inference parameters

Refer to `config/default_config.yaml` for a complete example.

## Project Structure

```
stable-diffusion-textual-inversion/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   ├── utils.py
│   └── inference.py
├── scripts/
│   ├── download_images.py
│   └── train_concept.py
├── config/
│   └── default_config.yaml
└── examples/
    └── example_inference.py
```

## Advanced Usage

### Customizing Templates

Modify the templates in the configuration file to create more diverse training prompts.

### Multi-Concept Training

While the current implementation supports single-concept training, the modular design allows for future expansion to multi-concept learning.



## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Textual Inversion Paper](https://textual-inversion.github.io/)





