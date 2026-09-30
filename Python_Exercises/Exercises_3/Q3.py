import os
import cv2
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter image path: ")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(image, kernel)
erosion = cv2.erode(image, kernel)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imwrite(os.path.join(output_dir, "dilation.jpg"), dilation)
cv2.imwrite(os.path.join(output_dir, "erosion.jpg"), erosion)
cv2.imwrite(os.path.join(output_dir, "opening.jpg"), opening)
cv2.imwrite(os.path.join(output_dir, "closing.jpg"), closing)

print("Morphological operations applied and saved")
