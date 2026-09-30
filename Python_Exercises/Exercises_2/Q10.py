import os
import cv2

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter image path: ")
image = cv2.imread(image_path)

blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(image, 100, 200)

cv2.imwrite(os.path.join(output_dir, "blurred.jpg"), blurred)
cv2.imwrite(os.path.join(output_dir, "edges.jpg"), edges)

print("Blurred and edge-detected images saved")
