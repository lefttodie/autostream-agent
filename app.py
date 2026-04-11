import streamlit as st
from agent_logic import agent_app
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AutoStream AI", page_icon="🎬")
st.title("AutoStream Intelligent Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_data" not in st.session_state:
    st.session_state.lead_data = {"name": None, "email": None, "platform": None}

for m in st.session_state.messages:
    with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
        st.markdown(m.content)

if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- CRITICAL: CAPTURE DATA FROM PREVIOUS QUESTION ---
    last_ai_text = ""
    if len(st.session_state.messages) > 1:
        for m in reversed(st.session_state.messages[:-1]):
            if isinstance(m, AIMessage):
                last_ai_text = m.content.lower()
                break

    # If the bot just asked for these, save the current prompt into lead_data
    if "full name?" in last_ai_text:
        st.session_state.lead_data["name"] = prompt
    elif "email?" in last_ai_text:
        st.session_state.lead_data["email"] = prompt
    elif "platform" in last_ai_text:
        st.session_state.lead_data["platform"] = prompt

    # --- INVOKE AGENT ---
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            inputs = {
                "messages": st.session_state.messages, 
                "lead_data": st.session_state.lead_data, 
                "context": ""
            }
            output = agent_app.invoke(inputs)
            
            # Update state with Agent's results
            st.session_state.lead_data = output.get("lead_data", st.session_state.lead_data)
            
            response = output["messages"][-1].content
            st.markdown(response)
            st.session_state.messages.append(AIMessage(content=response))