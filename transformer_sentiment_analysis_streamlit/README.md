# Sentiment Analysis App
This app analyzes the sentiment of a given text using a pre-trained DistilBERT model. It uses the Hugging Face Transformers library to load the pre-trained sentiment analysis model.

## Installation and Setup
To run this app, you will need to have streamlit and transformers Python packages installed. You can install them using pip:

```bash
pip install -r requirements.txt
```

To start the app, run the following command in your terminal:

```bash
streamlit run app.py

```
## How to use the app
1. Once the app is running, you will see the title "Sentiment Analysis App" and a brief description of what the app does.
2. Enter the text you want to analyze in the text area provided.
3. Click on the "Analyze" button to run the sentiment analysis on the text.
4. The app will display the result in JSON format, showing the predicted sentiment and its confidence score.
5. You can analyze another text by repeating steps 2-4.

## About the code
The app is written in Python using the streamlit and transformers packages. It loads a pre-trained sentiment analysis model from the Hugging Face Transformers library, and uses it to predict the sentiment of user input text. The app then displays the predicted sentiment and its confidence score in JSON format.