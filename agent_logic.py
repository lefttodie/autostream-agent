import os
from typing import Annotated, TypedDict, List, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# --- 1. FAST CONNECTION SETUP ---
# Standardizing the Embeddings call
_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

_vs = None

def get_vectorstore():
    global _vs
    if _vs is None:
        _vs = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX"), 
            embedding=_embeddings
        )
    return _vs

def get_llm():
    # Switching to a more stable model string
    return ChatOpenAI(
        model="openrouter/auto", 
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The chat history"]
    intent: str  
    lead_data: Dict[str, Optional[str]] 
    context: str 

def mock_lead_capture(name, email, platform):
    print(f"\n--- TOOL EXECUTION SUCCESS: {name} | {email} | {platform} ---")
    return f"✅ Success! I've registered you, {name}. Our team will contact you at {email}."

# --- 2. THE NODES ---

def classifier_node(state: AgentState):
    msg = state['messages'][-1].content.lower().strip()
    
    # 0ms Instant Bypass for Greetings
    if msg in ["hi", "hello", "hey", "hii", "hola"]:
        return {"intent": "greeting"}
    
    # Keyword Force for Inquiry (Plans/Pricing)
    if any(x in msg for x in ["plan", "price", "refund", "cost", "feature", "limit"]):
        return {"intent": "inquiry"}
        
    if any(x in msg for x in ["sign up", "join", "buy", "start", "register"]):
        return {"intent": "lead_gen"}
    
    try:
        llm = get_llm()
        prompt = f"Classify intent: '{msg}'. Categories: inquiry, lead_gen, greeting, out_of_scope. Return 1 word."
        intent = llm.invoke(prompt).content.strip().lower()
        return {"intent": intent}
    except:
        return {"intent": "inquiry"} # Fallback to inquiry if AI fails

def rag_node(state: AgentState):
    if state['intent'] == 'inquiry':
        try:
            vs = get_vectorstore()
            docs = vs.similarity_search(state['messages'][-1].content, k=2)
            return {"context": "\n".join([d.page_content for d in docs])}
        except:
            return {"context": ""}
    return {"context": ""}

def responder_node(state: AgentState):
    intent = state['intent']
    lead = state['lead_data']
    context = state.get('context', "")
    last_msg = state['messages'][-1].content.lower()

    if intent == "greeting":
        return {"messages": [AIMessage(content="Hi! I'm the AutoStream Assistant. Ask about our plans or say 'Sign me up' to start.")]}

    if context:
        llm = get_llm()
        ans = llm.invoke(f"Context: {context}\nQuestion: {last_msg}\nAnswer concisely.").content
        return {"messages": [AIMessage(content=ans)]}

    if intent == "lead_gen" or (lead and lead.get('name')):
        if not lead.get('name'):
            return {"messages": [AIMessage(content="I'd love to help! What is your full name?")]}
        if not lead.get('email'):
            return {"messages": [AIMessage(content="Got it. What is your email address?")]}
        if not lead.get('platform'):
            return {"messages": [AIMessage(content="Which platform do you create for (YouTube/Instagram)?")]}
        
        res = mock_lead_capture(lead['name'], lead['email'], lead['platform'])
        return {
            "messages": [AIMessage(content=res)], 
            "lead_data": {"name": None, "email": None, "platform": None}
        }

    if intent == "out_of_scope":
        return {"messages": [AIMessage(content="I'm here for AutoStream help only. I can't answer general questions like that.")]}

    return {"messages": [AIMessage(content="How else can I help you with AutoStream?")]}

# --- 3. THE GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("classify", classifier_node)
builder.add_node("retrieve", rag_node)
builder.add_node("respond", responder_node)
builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "respond")
builder.add_edge("respond", END)
agent_app = builder.compile()