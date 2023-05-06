# Import Python libraries
import streamlit as st
from streamlit_chat import message
import openai
import os

# Define the model engine and your OpenAI API key
model_engine = "text-davinci-003"

# Follow README.md to create an Open API secret key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define Streamlit web page's configuration
st.set_page_config(
    page_title="ChatGPT-based Chat Application",
    page_icon="assets/wcd_logo_new_1.png",
)

# Add description on the website
st.image('assets/wcd_logo_new_2.png', width=100)
st.title("ChatGPT-based Chat Application")
st.subheader("Description")
st.info(
    '''This is a Streamlit web application that allows you to interact with 
    the OpenAI API's implementation of the ChatGPT model.
    \nPlease enter a **query** in the **text box** and **press enter** to receive 
    a **response** from the ChatGPT.
    '''
)

# Store the chat history in one session
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_user_query():
    """
    This function allows user to type a query in a textbox and extract the text
    """

    # Text box for user query
    st.write("This is the chat interface")
    query = st.text_input("Hello!!!! Please enter your query here. e.g. What is Python?")

    return query


def call_openai_chat_gpt(user_query):
    """
    This functions call OpenAI API and receive a response
    """
    
    # Use OpenAI API with the following parameters
    response = openai.Completion.create(
        engine=model_engine,
        prompt=user_query,
        max_tokens=1024,
        n=1,
        temperature=0.5,
    )

    print(response)
    # Retrieve the response text
    text = response.choices[0].text

    # Return the text
    return text


def main():
    """
    This function allows user to type a query in a textbox and send the query to OpenAI to receive a response from ChatGPT
    """
    # Get user query as text input
    query = get_user_query()

    if query:
        # Pass the user input query to the ChatGPT function
        output = call_openai_chat_gpt(query)
        # store the output
        st.session_state.past.append(query)
        st.session_state.generated.append(output)

    # Render chat history
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


# Call the main function to start the chat application
main()
