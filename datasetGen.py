import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import os

# Create a directory for the dataset
output_dir = 'inverted_alphabet_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the alphabets and image properties
alphabets = ['U', 'A', 'J', 'F', 'R']
num_images = 1000
image_size = (64, 64)
font_size = 50

# Load a font (adjust the path to the font file as needed)
font = ImageFont.truetype("FiraCode-Bold.ttf", font_size)

for i in range(num_images):
    # Create a new image with white background
    image = Image.new('L', image_size, color='white')
    draw = ImageDraw.Draw(image)

    # Choose a random alphabet and a random position
    char = random.choice(alphabets)
    position = (random.randint(0, image_size[0] - font_size), random.randint(0, image_size[1] - font_size))

    # Draw the character with a random shade of gray/black
    shade = random.randint(0, 50)
    draw.text(position, char, font=font, fill=shade)

    # Invert the image
    inverted_image = ImageOps.flip(image)

    # Save the image
    image_path = os.path.join(output_dir, f'{i:04d}_{char}.png')
    inverted_image.save(image_path)

print(f"Dataset of {num_images} images generated in '{output_dir}' directory.")
