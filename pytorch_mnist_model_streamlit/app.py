import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# Define CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Load pre-trained model
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=torch.device('cpu')))
model.eval()

# Function to preprocess and predict the image class
def predict_digit(img):
    # Preprocess image
    img = img.resize((28, 28), Image.ANTIALIAS).convert('L')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    # Predict image class using the pre-trained model
    with torch.no_grad():
        output = model(img)
    return output.argmax().item()

# Define Streamlit web page configuration
st.set_page_config(
    page_title="MNIST Handwritten Digit Classification App",
    page_icon="✍️",
)

# Add title and description
st.title("MNIST Handwritten Digit Classification App")
st.write("This app classifies handwritten digits based on a pre-trained CNN model using PyTorch.")

# Add canvas for drawing the digit
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=20,
    stroke_color="rgba(255, 255, 255, 1)",
    background_color="rgba(0, 0, 0, 1)",
    width=280,
    height=280,
    drawing_mode="freedraw",
)

# If a digit is drawn on the canvas
if canvas_result.image_data is not None:
    # Display the drawn image
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)[:, :, :3])
    st.image(img, caption="Drawn Digit", use_column_width=True)

    # Predict the digit and display the result
    with st.spinner("Classifying the digit..."):
      predicted_digit = predict_digit(img)
    st.success(f"Predicted Digit: {predicted_digit}")