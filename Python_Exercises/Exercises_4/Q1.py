import os
from ultralytics import YOLO

output_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolov8n.pt")

image_path = input("Enter image path: ")
results = model(image_path)

output_path = os.path.join(output_dir, "detected.jpg")
results[0].show()
results[0].save(output_path)

print("Detection complete, saved to", output_path)
