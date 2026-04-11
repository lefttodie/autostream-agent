import streamlit as st
from agent_logic import agent_app
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AutoStream AI", page_icon="🎬")
st.title(" AutoStream Intelligent Agent")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_data" not in st.session_state:
    st.session_state.lead_data = {"name": None, "email": None, "platform": None}

# Display Messages
for m in st.session_state.messages:
    with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
        st.markdown(m.content)

# User Input
if prompt := st.chat_input("Ask about AutoStream plans..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simple Lead Data Parsing based on last Assistant message
    last_ai_text = ""
    if len(st.session_state.messages) > 1:
        for m in reversed(st.session_state.messages[:-1]):
            if isinstance(m, AIMessage):
                last_ai_text = m.content
                break

    lead = st.session_state.lead_data
    if "full name?" in last_ai_text:
        lead["name"] = prompt
    elif "email address?" in last_ai_text:
        lead["email"] = prompt
    elif "Which platform" in last_ai_text:
        lead["platform"] = prompt

    # Call the Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            inputs = {
                "messages": st.session_state.messages, 
                "lead_data": lead, 
                "context": ""
            }
            output = agent_app.invoke(inputs)
            
            # Update Lead Data from Agent's choice
            if "lead_data" in output:
                st.session_state.lead_data = output["lead_data"]
            
            response = output["messages"][-1].content
            st.markdown(response)
            st.session_state.messages.append(AIMessage(content=response))