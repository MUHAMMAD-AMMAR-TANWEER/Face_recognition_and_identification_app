import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

y_labels = []
x_train = []


for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(path, label)
            pil_image = Image.open(path).convert(
                "L"
            )  # this converts Image into grayscale
            image_array = np.array(
                pil_image, "uint8"
            )  # use uint8 because 8 bit makes 255 0-255 positive numbers
            print(image_array)
