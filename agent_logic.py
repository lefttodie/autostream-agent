import os
from typing import Annotated, TypedDict, List, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

def get_llm():
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

def classifier_node(state: AgentState):
    llm = get_llm()
    msg = state['messages'][-1].content
    
    last_ai_msg = ""
    if len(state['messages']) > 1:
        for m in reversed(state['messages'][:-1]):
            if isinstance(m, AIMessage):
                last_ai_msg = m.content.lower()
                break
    
    # Context Lock for Lead Gen
    if "full name?" in last_ai_msg or "email?" in last_ai_msg or "platform" in last_ai_msg:
        return {"intent": "lead_gen"}

    system_prompt = f"""
    You are the Router for AutoStream. Analyze the user's message.
    USER MESSAGE: "{msg}"

    RULES:
    1. User wants to sign up/join/register -> return 'lead_gen'.
    2. User asks about prices, plans, or comparisons -> return 'inquiry'.
    3. User greets -> return 'greeting'.
    4. Unrelated/General topics -> return 'out_of_scope'.

    OUTPUT ONLY ONE WORD: inquiry | lead_gen | greeting | out_of_scope
    """
    intent = llm.invoke(system_prompt).content.strip().lower()
    return {"intent": intent if intent in ["inquiry", "lead_gen", "greeting", "out_of_scope"] else "inquiry"}

def rag_node(state: AgentState):
    if state['intent'] == 'inquiry':
        vs = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX"), embedding=_embeddings)
        docs = vs.similarity_search(state['messages'][-1].content, k=2)
        return {"context": "\n".join([d.page_content for d in docs])}
    return {"context": ""}

def responder_node(state: AgentState):
    intent = state['intent']
    lead = state['lead_data']
    context = state.get('context', "")
    llm = get_llm()

    # Dynamic Suffix for interaction
    suffix = "\n\n---\n*Would you like to know more about our pricing plans or how AutoStream can speed up your workflow?*"

    if intent == "inquiry":
        prompt = f"Context: {context}\nUser: {state['messages'][-1].content}\nAnswer professionally with bold plan names and $ prices. End by asking if they want to see a specific feature list."
        res = llm.invoke(prompt).content
        return {"messages": [AIMessage(content=res + suffix)]}

    if intent == "lead_gen":
        if not lead.get('name'):
            return {"messages": [AIMessage(content="I can get you started! What is your full name?")]}
        if not lead.get('email'):
            return {"messages": [AIMessage(content=f"Thanks {lead['name']}, what is your email?")]}
        if not lead.get('platform'):
            return {"messages": [AIMessage(content="And which platform (YouTube/Instagram)?")]}
        
        # Professional Success Message
        success_msg = f"""
### 🎉 Success! 

Thank you, **{lead['name']}**. We have received your details and our team will reach out to **{lead['email']}** within the next 24 hours to finalize your setup on {lead['platform']}.

*Is there anything else I can help you with regarding our features?*
        """
        return {
            "messages": [AIMessage(content=success_msg)],
            "lead_data": {"name": None, "email": None, "platform": None}
        }

    if intent == "out_of_scope":
        return {"messages": [AIMessage(content="I'm here to help you automate your video content with AutoStream! I can't answer general questions, but I can tell you all about our AI tools. Should we look at the pricing?")]}

    if intent == "greeting":
        return {"messages": [AIMessage(content="Hello! I'm the AutoStream assistant. Ready to turn your ideas into viral videos? Would you like to see our plans or sign up for an account?")]}

    return {"messages": [AIMessage(content="I'm here to help. You can ask about our plans or say 'sign up' to begin.")]}

# Graph remains same
builder = StateGraph(AgentState)
builder.add_node("classify", classifier_node)
builder.add_node("retrieve", rag_node)
builder.add_node("respond", responder_node)
builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "respond")
builder.add_edge("respond", END)
agent_app = builder.compile()