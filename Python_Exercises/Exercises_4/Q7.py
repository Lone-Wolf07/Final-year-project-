import os
import cv2

output_dir = os.path.dirname(os.path.abspath(__file__))

image_path = input("Enter damaged image path: ")
mask_path = input("Enter mask image path: ")

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite(os.path.join(output_dir, "inpainted.jpg"), inpainted)
print("Inpainted image saved")
