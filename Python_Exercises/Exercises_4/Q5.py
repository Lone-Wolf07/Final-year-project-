import pytesseract
import cv2

image_path = input("Enter image path: ")
image = cv2.imread(image_path)

text = pytesseract.image_to_string(image)
print("Extracted text:")
print(text)
