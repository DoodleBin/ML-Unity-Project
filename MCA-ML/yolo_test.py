from ultralytics import YOLO
from tcp_service import detect_objects
import json

# Test image preprocessing
img_path = 'C:/Users/alexah1/Documents/GitHub/ML-Unity-Project/MCA-ML/data/zoo.jpg'

#turn byte string into image, then resize image
#img_byte = string_to_image(byte_string)
img = resize(img_path)

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
boxesJson = detect_objects(img, model)  # Predict on an image
with open("boundingBoxInfo.json", 'w', encoding="utf-8") as f:
    json.dump(boxesJson, f, indent=2)

results = model(img_path)
results[0].show()
print(boxesJson)

#path = model.export(format="onnx")  # Returns the path to the exported model