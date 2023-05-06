# Object Detection App

This is a Streamlit web application that detects objects in an uploaded image using a pre-trained Faster R-CNN model with a ResNet-50 backbone. The app draws bounding boxes and labels around the detected objects in the image.

## Installation

To run this app, you need to have Python and the following packages installed:

- streamlit
- torch
- torchvision
- pillow
- numpy
- opencv-python

You can install these packages by running the following command:

```bash
pip install streamlit torch torchvision pillow numpy opencv-python
```

or by running the following command that uses the provided requirements.txt file
```bash
pip install -r requirements.txt
```

## Usage

To run the app, execute the following command:

```bash
streamlit run app.py
```

This will start a local server and open the app in your web browser.

To use the app, click the "Upload an image" button and select an image file (in JPG, JPEG, or PNG format) to upload. The app will detect objects in the image and draw bounding boxes and labels around them.

## Credits

The app uses a pre-trained Faster R-CNN model with a ResNet-50 backbone from the torchvision library, and the COCO dataset category names from the Detectron2 library.

