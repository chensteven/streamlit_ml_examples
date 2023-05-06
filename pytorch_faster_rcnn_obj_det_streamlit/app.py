import streamlit as st
import torch
import torchvision
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import cv2

# Load pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to preprocess and predict the objects in the image
def detect_objects(img):
    # Preprocess image
    img_tensor = ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)

    # Predict objects using the pre-trained model
    with torch.no_grad():
        predictions_list = model(img_tensor)

    # Get the predictions dictionary from the list
    predictions = predictions_list[0]

    # Filter predictions with confidence scores above the threshold
    threshold = 0.5
    filtered_preds = [
        {
            "boxes": predictions["boxes"][i].tolist(),
            "labels": int(predictions["labels"][i].item()),
            "scores": float(predictions["scores"][i].item()),
        }
        for i in range(predictions["labels"].size(0))
        if predictions["scores"][i] > threshold
    ]

    return filtered_preds

# Define Streamlit web page configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
)

# Add title and description
st.title("Object Detection App")
st.write("This app detects objects in an uploaded image using a pre-trained Faster R-CNN model with a ResNet-50 backbone.")

# Load COCO category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, predictions):
    img = np.array(image)
    for pred in predictions:
        # Get box coordinates and label
        x1, y1, x2, y2 = map(int, pred["boxes"])
        label_index = int(pred["labels"])
        label = COCO_INSTANCE_CATEGORY_NAMES[label_index]

        # Draw bounding box and label
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img


# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Detect objects in the image
    predictions = detect_objects(image)

    # Draw bounding boxes and labels on the image
    result_img = draw_boxes(image, predictions)

    # Display the result
    st.image(result_img, caption="Image with detected objects", use_column_width=True)
