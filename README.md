# Stable Diffusion Textual Inversion

## Overview

This project provides a comprehensive toolkit for training and using Textual Inversion with Stable Diffusion models. Textual Inversion is a technique that allows you to teach a Stable Diffusion model new concepts using just a few example images.

![Textual Inversion Concept](https://textual-inversion.github.io/static/images/editing/colorful_teapot.JPG)

## Features

- 🖼️ Train custom concepts with just 3-5 images
- 🤖 Support for both object and style learning
- 📊 Flexible configuration options
- 🚀 Easy-to-use inference pipeline
- 🔧 Modular and extensible design

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended
- PyTorch
- Hugging Face Diffusers library

### Clone the Repository

```bash
git clone https://github.com/yourusername/stable-diffusion-textual-inversion.git
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

## Troubleshooting

- Ensure you have a CUDA-compatible GPU
- Check that all dependencies are correctly installed
- Verify image format and quality of training images

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Textual Inversion Paper](https://textual-inversion.github.io/)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{textual_inversion_toolkit,
  title = {Stable Diffusion Textual Inversion Toolkit},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/stable-diffusion-textual-inversion}
}
```

