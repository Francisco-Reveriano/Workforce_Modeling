# Import Libraries
from galileo import GalileoLogger
from galileo.handlers.openai_agents import GalileoTracingProcessor
from agents import set_trace_processors
import streamlit as st
import asyncio
from agents import Runner, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
import os
import toml
import warnings
warnings.filterwarnings("ignore")

# Setup environment
secrets = toml.load(".streamlit/secrets.toml")
for key, value in secrets.items():
    os.environ[key] = str(value)

# Import Agents
set_trace_processors([GalileoTracingProcessor()])
from src.Agents.Basic_QA_Agent import Basic_QA_Agent
from src.Agents.GenAI_Use_Case_Agent import GenAI_Use_Case_Agent

session = SQLiteSession("user_123")

st.set_page_config(page_title="Workforce_Planning_Agent", page_icon="❓", layout="centered")
st.title("Workforce_Planning_Agent")

# Sidebar controls
with st.sidebar:
    st.markdown("Workforce Planning Agents")
    st.header("Controls")
    agent_choice = st.selectbox("Select agent", ["GenAI_Use_Case_Agent", "Basic_QA_Agent"], index=0)
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
                selected_agent = Basic_QA_Agent if agent_choice == "Basic_QA_Agent" else GenAI_Use_Case_Agent
                placeholder = st.empty()
                collected = [""]

                async def stream():
                    result = Runner.run_streamed(selected_agent, input=prompt, session=session)
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            collected[0] += event.data.delta
                            placeholder.markdown(collected[0])

                asyncio.run(stream())
                output_text = collected[0]
            except Exception as e:
                output_text = f"Error: {e}"
                st.markdown(output_text)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": output_text})

