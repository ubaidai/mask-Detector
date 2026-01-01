import gradio as gr
from ultralytics import YOLO
import numpy as np

# Load trained model
model = YOLO("best.pt")

def object_detection(image):
    results = model.predict(image, imgsz=640)
    output = results[0].plot()
    return output

app = gr.Interface(
    fn=object_detection,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detected Output"),
    title="Mask Detection System",
    description="YOLO model trained on Roboflow dataset and deployed using Gradio"
)

app.launch()
