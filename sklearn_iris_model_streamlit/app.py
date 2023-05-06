import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Define Streamlit web page configuration
st.set_page_config(
    page_title="Iris Flower Prediction App",
    page_icon="🌸",
)

# Add title and description
st.title("Iris Flower Prediction App")
st.write("This app predicts the class of Iris flowers based on their features.")

# Add feature input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 2.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.5, step=0.1)

# Arrange feature inputs in a DataFrame
features = pd.DataFrame(
    {
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width],
    }
)

# Display user inputs
st.subheader("User Inputs")
st.write(features)

# Predict the class of the Iris flower
prediction = clf.predict(features)

# Display the predicted class
st.subheader("Prediction")
st.write(iris.target_names[prediction][0])
