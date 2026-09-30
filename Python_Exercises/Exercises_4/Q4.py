import os
import cv2
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter image path: ")
image = cv2.imread(image_path)
pixels = image.reshape((-1, 3)).astype(np.float32)

k = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented = centers[labels.flatten()]
segmented = segmented.reshape(image.shape)

cv2.imwrite(os.path.join(output_dir, "segmented.jpg"), segmented)
print("Segmented image saved")
