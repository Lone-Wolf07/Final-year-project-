import os
import cv2
import pywt
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter noisy image path: ")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

coeffs = pywt.wavedec2(image, "db1", level=2)
coeffs_thresh = list(coeffs)
coeffs_thresh[1:] = [tuple(pywt.threshold(c, value=20, mode="soft") for c in level) for level in coeffs[1:]]

denoised = pywt.waverec2(coeffs_thresh, "db1")
denoised = np.clip(denoised, 0, 255).astype(np.uint8)

cv2.imwrite(os.path.join(output_dir, "denoised.jpg"), denoised)
print("Denoised image saved")
