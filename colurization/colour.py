import torch
from pathlib import Path
import os

from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer

# Fix for PyTorch >=2.6
original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, *args, **kwargs)
torch.load = patched_load

device.set(device=DeviceId.GPU0)  # Change to CPU if needed

colorizer = get_image_colorizer(artistic=True)

def colorize_image(input_path, render_factor=35, output_path=None):
    if output_path is None:
        output_path = f'result_images/{Path(input_path).name}'
    os.makedirs('result_images', exist_ok=True)
    colorizer.plot_transformed_image(
        path=input_path,
        render_factor=render_factor,
        display_render_factor=True,
        figsize=(8,8)
    )
    print(f"Saved result to: {output_path}")
    return output_path
