import os
import cv2
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

def embed_watermark(image_path, watermark_path, output_path):
    image = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))
    watermark_bit = (watermark > 127).astype(np.uint8)

    watermarked = image & 0xFE
    watermarked[:, :, 0] |= watermark_bit

    cv2.imwrite(output_path, watermarked)

def extract_watermark(image_path, output_path):
    image = cv2.imread(image_path)
    extracted = (image[:, :, 0] & 1) * 255
    cv2.imwrite(output_path, extracted)

image_path = input("Enter image path: ")
watermark_path = input("Enter watermark image path: ")

watermarked_path = os.path.join(output_dir, "watermarked.jpg")
extracted_path = os.path.join(output_dir, "extracted_watermark.jpg")

embed_watermark(image_path, watermark_path, watermarked_path)
extract_watermark(watermarked_path, extracted_path)

print("Watermark embedded and extracted")
