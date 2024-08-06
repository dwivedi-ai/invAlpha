import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import random
import os

# Create a directory for the dataset
output_dir = 'inverted_alphabet_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the alphabets and image properties
alphabets = ['U', 'A', 'J', 'F', 'R']
num_images = 1000
image_size = (128, 128)  # Increased image size for better quality
font_size = 80  # Increased font size

# Load a font (adjust the path to the font file as needed)
font = ImageFont.truetype("FiraCode-Bold.ttf", font_size)

for i in range(num_images):
    # Create a new image with white background
    image = Image.new('L', image_size, color='white')
    draw = ImageDraw.Draw(image)

    # Choose a random alphabet
    char = random.choice(alphabets)

    # Get the size of the character
    left, top, right, bottom = font.getbbox(char)
    char_width = right - left
    char_height = bottom - top

    # Calculate position to center the character
    position = ((image_size[0] - char_width) // 2, (image_size[1] - char_height) // 2)

    # Draw the character with a random shade of gray/black
    shade = random.randint(0, 50)
    draw.text(position, char, font=font, fill=shade)

    # Add some noise to the image
    noise = np.random.normal(0, 2, image_size)
    noisy_image = np.array(image) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    image = Image.fromarray(noisy_image)

    # Apply a slight Gaussian blur for smoother edges
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Invert the image
    inverted_image = ImageOps.invert(image)

    rotation = random.choice([0, 90, 180, 270])
    inverted_image = inverted_image.rotate(rotation, resample=Image.BICUBIC, expand=False)

    # Save the image
    image_path = os.path.join(output_dir, f'{i:04d}_{char}.png')
    inverted_image.save(image_path, quality=95)  # Increased save quality

print(f"Dataset of {num_images} images generated in '{output_dir}' directory.")