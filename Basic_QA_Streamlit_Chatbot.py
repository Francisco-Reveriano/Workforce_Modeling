from src.Agents.Basic_QA_Agent import Basic_QA_Agent
import streamlit as st
import asyncio
from agents import Runner, SQLiteSession

session = SQLiteSession("user_123")

st.set_page_config(page_title="Basic Q&A Chatbot", page_icon="❓", layout="centered")
st.title("Basic Q&A Chatbot")

# Sidebar controls
with st.sidebar:
    st.markdown("Simple Chatbot Connected to OpenAI Assistants")
    st.header("Controls")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask a question")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(Runner.run(Basic_QA_Agent, prompt, session=session))
                output_text = getattr(result, "final_output", str(result))
            except Exception as e:
                output_text = f"Error: {e}"
            st.markdown(output_text)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": output_text})

