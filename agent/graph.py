from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.nodes import *

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", classify_intent)
    graph.add_node("greet", handle_greeting)
    graph.add_node("rag", handle_rag)
    graph.add_node("lead", handle_lead)

    graph.set_entry_point("intent")

    def route(state):
        if state["intent"] == "greeting":
            return "greet"
        elif state["intent"] == "product":
            return "rag"
        else:
            return "lead"

    graph.add_conditional_edges("intent", route)

    return graph.compile()