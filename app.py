import streamlit as st
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Rancho Chat App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image_url"):
            st.image(message["image_url"])

# Get user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to the backend
    try:
        response = requests.post("http://localhost:8000/chat", json={"user_input": prompt, "context": ""})
        response.raise_for_status()
        chat_response = response.json()
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": chat_response["response"], "image_url": chat_response.get("image_url")})
        with st.chat_message("assistant"):
            st.markdown(chat_response["response"])
            if chat_response.get("image_url"):
                st.image(chat_response["image_url"])
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
        logger.error(f"Error communicating with the backend: {e}")
