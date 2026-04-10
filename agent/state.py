from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    intent: str
    name: str
    email: str
    platform: str