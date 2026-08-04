import streamlit as st
import time
from chatbot import create_chatbot
import ui

def check_rate_limit():

    MAX_REQUESTS = 10
    TIME_WINDOW = 60  # seconds

    if "request_times" not in st.session_state:
        st.session_state.request_times = []

    current_time = time.time()

    st.session_state.request_times = [
        t for t in st.session_state.request_times
        if current_time - t < TIME_WINDOW
    ]

    if len(st.session_state.request_times) >= MAX_REQUESTS:
        return False

    st.session_state.request_times.append(current_time)

    return True


st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.load_css()


@st.cache_resource(show_spinner=False)
def load_chatbot():
    return create_chatbot()

with st.spinner("Loading Medical RAG Assistant... Please wait."):
    st.markdown(
        """
        <style>
        .stSpinner > div > div {
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    chatbot = load_chatbot()


ui.render_sidebar()
ui.render_header("Medical RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    ui.render_message(message["role"], message["content"])


if prompt := ui.render_chat_input("Type your medical question..."):

    if not check_rate_limit():
        ui.render_message(
            "assistant",
            "Too many requests. Please wait a minute before trying again."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    ui.render_message("user", prompt)

    with st.spinner("Searching knowledge base..."):
        st.markdown(
        """
        <style>
        .stSpinner > div > div {
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )
        history = ""

        for msg in st.session_state.messages[-6:]:  # last 3 turns
            history += f"{msg['role'].capitalize()}: {msg['content']}\n"

        response = chatbot(prompt, history)

    ui.render_message("assistant", response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
