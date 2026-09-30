import os
import cv2

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter image path: ")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

blurred = cv2.GaussianBlur(image, (3, 3), 0)
log = cv2.Laplacian(blurred, cv2.CV_64F)

cv2.imwrite(os.path.join(output_dir, "sobel.jpg"), sobel)
cv2.imwrite(os.path.join(output_dir, "laplacian.jpg"), laplacian)
cv2.imwrite(os.path.join(output_dir, "log.jpg"), log)

print("Edge detected images saved")
