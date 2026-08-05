import html
from pathlib import Path
import markdown as _markdown
import streamlit as st

CSS_PATH = Path(__file__).parent / "style.css"

def load_css():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    """
    Render the blue sidebar.

    Contains only the logo/title and the About section, per the reference
    design — no extra buttons, metrics, or widgets.
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-badge">+</div>
            </div>
            <h1 class="sidebar-title">Medical RAG<br>Assistant</h1>
            <hr class="sidebar-divider" />
            <div class="sidebar-about-heading">
                <span class="sidebar-about-icon">&#9432;</span>
                <span>About</span>
            </div>
            <p class="sidebar-about-text">
                Medical RAG Assistant provides
                evidence-grounded answers using
                trusted MedlinePlus medical
                knowledge.
                This assistant is designed for
                educational purposes and is not
                a substitute for professional
                medical advice.
            </p>
            """,
            unsafe_allow_html=True,
        )


def render_header(title: str = "Medical RAG Assistant") -> None:
    st.markdown(
        f"""
        <h1 class="main-title">
            {html.escape(title)}
        </h1>
        """,
        unsafe_allow_html=True,
    )


def _render_content_html(text: str) -> str:
    """
    Convert message text into safe HTML for the bubble.

    Streamlit's own st.markdown will not reliably parse markdown syntax
    once it's nested inside a raw HTML <div> (which our bubble needs, to
    get the inline timestamp and rounded "tail" corner). So instead we:

    1. HTML-escape the raw text first, so a user question or model output
       containing "<" / ">" / "&" can never inject markup.
    2. Run the escaped text through the `markdown` library — the same
       package Streamlit itself depends on — so **bold**, bullet lists,
       etc. from the chatbot's answers render as real HTML rather than
       showing up as literal asterisks.
    """
    escaped = html.escape(text)
    return _markdown.markdown(escaped, extensions=["nl2br", "sane_lists"])


def render_message(role: str, content: str) -> None:
    """
    Render one chat bubble.

    Parameters
    ----------
    role: "user" or "assistant"
    content: the message text (already the final chatbot/user text —
        no chatbot logic happens here)
    timestamp: optional pre-set label; defaults to the current time so
        each message gets stamped once, at render time.

    st.chat_message is used only as an accessible/semantic container —
    the visible bubble is custom markup styled in style.css so it can match
    the reference design (no avatar, inline timestamp, flattened "tail"
    corner).
    """
    safe_content = _render_content_html(content)
    bubble_class = "bubble-user" if role == "user" else "bubble-assistant"
    row_class = "message-row-user" if role == "user" else "message-row-assistant"

    with st.chat_message(role):
        st.markdown(
            f"""
            <div class="{row_class}">
                <div class="chat-bubble {bubble_class}">
                    <div class="bubble-content">{safe_content}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_input(placeholder: str = "Type your medical question..."):
    """Thin wrapper around st.chat_input so app.py doesn't hardcode copy."""
    return st.chat_input(placeholder)
