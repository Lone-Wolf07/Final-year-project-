import os
import cv2
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter image path: ")
image = cv2.imread(image_path)
rows, cols = image.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))

translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(image, translation_matrix, (cols, rows))

flipped = cv2.flip(image, 1)

scaled = cv2.resize(image, None, fx=1.5, fy=1.5)

shear_matrix = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
sheared = cv2.warpAffine(image, shear_matrix, (cols, rows))

cv2.imwrite(os.path.join(output_dir, "rotated.jpg"), rotated)
cv2.imwrite(os.path.join(output_dir, "translated.jpg"), translated)
cv2.imwrite(os.path.join(output_dir, "flipped.jpg"), flipped)
cv2.imwrite(os.path.join(output_dir, "scaled.jpg"), scaled)
cv2.imwrite(os.path.join(output_dir, "sheared.jpg"), sheared)

print("Augmented images saved")
