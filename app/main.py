import streamlit as st
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import build_graph

graph = build_graph()

# 🔹 Page config
st.set_page_config(page_title="AutoStream AI", page_icon="🤖", layout="wide")

st.title("AutoStream AI Agent")

# 🔹 Initialize session state
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "intent": "",
        "name": "",
        "email": "",
        "platform": ""
    }

# 🔹 Chat history UI
for msg in st.session_state.state["messages"]:
    role = "user" if msg == st.session_state.state["messages"][-1] else "assistant"
    with st.chat_message(role):
        st.markdown(msg)

# 🔹 Chat input (bottom like ChatGPT)
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user input to state
    st.session_state.state["messages"].append(user_input)

    # Invoke graph
    result = graph.invoke(st.session_state.state)

    response = result["messages"][-1]

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save response
    st.session_state.state["messages"].append(response)

    # 🔹 Your extraction logic (unchanged)
    if "@" in user_input:
        st.session_state.state["email"] = user_input
    elif "youtube" in user_input.lower():
        st.session_state.state["platform"] = "YouTube"
    else:
        st.session_state.state["name"] = user_input