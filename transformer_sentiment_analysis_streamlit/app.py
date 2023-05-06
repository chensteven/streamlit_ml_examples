import streamlit as st
from transformers import pipeline

# Load pre-trained sentiment analysis model
nlp = pipeline("sentiment-analysis")

# Define Streamlit web page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💭",
)

# Add title and description
st.title("Sentiment Analysis App")
st.write("This app analyzes the sentiment of a given text using a pre-trained DistilBERT model.")

# User input text
user_input = st.text_area("Enter your text here:")

if user_input:
    # Predict sentiment using the pre-trained model
    sentiment_result = nlp(user_input)

    print(sentiment_result)

    # Display the result
    st.write("Sentiment Analysis Result:")
    st.json(sentiment_result)
