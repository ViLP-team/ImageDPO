import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def corrupt_image(image_path, output_path, probabilities, severity):
    # Open the image
    image = Image.open(image_path)

    # Define the corruption methods
    corruption_methods = [add_noise, blur_image, pixelate_image, invert_colors]

    # Apply each corruption method based on the given probability
    for method, prob in zip(corruption_methods, probabilities):
        if random.random() < prob:
            image = method(image, severity)

    # Save the corrupted image
    image.save(output_path)


def add_noise(image, severity):
    # Convert image to numpy array
    np_image = np.array(image)
    # Generate random noise
    noise = np.random.normal(0, severity, np_image.shape)
    # Add noise to the image
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    # Convert numpy array back to image
    return Image.fromarray(np_image)


def blur_image(image, severity):
    return image.filter(ImageFilter.GaussianBlur(radius=severity))


def pixelate_image(image, severity):
    # Resize down
    small = image.resize(
        (image.width // severity, image.height // severity), resample=Image.BILINEAR
    )
    # Resize up
    return small.resize(image.size, Image.NEAREST)


def invert_colors(image, severity=None):
    return ImageOps.invert(image)
