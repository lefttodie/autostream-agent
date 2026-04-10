from tools.lead_capture import mock_lead_capture
from rag.retriever import get_retriever
from llm.llm import get_llm

llm = get_llm()
retriever = get_retriever()

def classify_intent(state):
    query = state["messages"][-1]

    if any(x in query.lower() for x in ["hi", "hello"]):
        state["intent"] = "greeting"
    elif any(x in query.lower() for x in ["price", "cost"]):
        state["intent"] = "product"
    elif any(x in query.lower() for x in ["buy", "start", "subscribe"]):
        state["intent"] = "high_intent"
    else:
        state["intent"] = "product"

    return state


def handle_greeting(state):
    return {"messages": ["Hello! How can I assist you today?"]}

def handle_rag(state):
    query = state["messages"][-1]

    if isinstance(query, dict):
        query = query.get("content", "")

    retriever = get_retriever()

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    llm = get_llm()

    response = llm.invoke(f"""
    Answer based on context:

    {context}

    Question: {query}
    """)

    state["messages"].append(response.content)

    return state

def handle_lead(state):
    if not state.get("name"):
        return {"messages": ["Please provide your name."]}

    elif not state.get("email"):
        return {"messages": ["Please provide your email."]}

    elif not state.get("platform"):
        return {"messages": ["Which platform do you use?"]}

    else:
        mock_lead_capture(
            state["name"],
            state["email"],
            state["platform"]
        )
        return {"messages": ["Thanks! We've captured your details."]}