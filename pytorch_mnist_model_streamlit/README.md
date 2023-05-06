# MNIST Handwritten Digit Classifier
This repository contains a PyTorch implementation of a convolutional neural network (CNN) that can classify handwritten digits from the MNIST dataset. The trained model can be used to predict the digit represented in a given image.

## Requirements
- streamlit
- torch
- torchvision
- Pillow
- numpy
- streamlit_drawable_canvas

To install the required packages, run the following command:
```bash
pip install streamlit torch torchvision Pillow numpy streamlit_drawable_canvas
```

or by running the following command that uses the provided requirements.txt file
```bash
pip install -r requirements.txt
```

## Usage
### Training the model
To train the CNN model, run the following command:

```bash
python train.py
```
This will train the model for 10 epochs on the MNIST dataset and save the trained model to mnist_cnn.pt.

### Running the app
To run the app, open a terminal or command prompt and run the following command:

```bash
streamlit run app.py
```
Once the app is running, it can be accessed in a web browser at the URL shown in the terminal.

## App Overview
The app is designed to allow users to draw a digit on a canvas and have the pre-trained CNN model predict the digit that was drawn.

The app's main page includes the following:

- A title that describes the app's purpose
- A description that explains how the app works
- A canvas that allows users to draw a digit

When the user draws a digit on the canvas, the following happens:

- The drawn digit is preprocessed and converted into a format that can be input to the pre-trained model
- The pre-trained model predicts the class of the drawn digit
- The predicted digit is displayed to the user

## Model
The model used in this app is a CNN that has been pre-trained on the MNIST dataset of handwritten digits. The CNN model is defined in the script and loaded from the mnist_cnn.pt file.

## Canvas
The canvas is implemented using the streamlit_drawable_canvas package. It allows the user to draw a digit using their mouse or touchpad. The size of the canvas and the stroke width and color can be configured in the st_canvas function.

## Prediction
When the user draws a digit on the canvas, the digit is preprocessed and converted into a format that can be input to the pre-trained model. The pre-trained model then predicts the class of the drawn digit using the predict_digit function. Finally, the predicted digit is displayed to the user.

## Credits
This implementation is based on the Streamlit app developed by Streamlit and can be found in their documentation. The CNN architecture used in the model is also based on their implementation.