import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'colurization'))



from colurization.colour import colorize_image


input_image_path = '/Users/bappa123/Desktop/Personalize-image-editing/test1.png'

colorize_image(input_image_path, render_factor=35)
