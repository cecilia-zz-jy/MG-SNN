import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from PIL import Image

def save_gray_image(data,  filename, output_dir, index):
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    normalized_data = (normalized_data * 255).astype(np.uint8)
    img = Image.fromarray(normalized_data)
    # img = img.resize(output_size, Image.ANTIALIAS)
    gray_img = img.convert('L')
    inverted_gray_img = Image.fromarray(255 - np.array(gray_img))
    indexed_filename = f"{filename}-{str(index+1).zfill(6)}.png"
    inverted_gray_img.save(os.path.join(output_dir, indexed_filename))
    print("图像已保存为:", indexed_filename)
