from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

class WorkflowState(TypedDict):
    user_prompt: str
    search_query: Optional[str]
    keywords: List[str]
    is_valid: bool
    invalid_reason: Optional[str]
    search_results: List[str]


def get_input_node(state: WorkflowState) -> WorkflowState:
    return state

def analyze_prompt_node(state: WorkflowState) -> WorkflowState:
    return state

def hybrid_search_node(state: WorkflowState) -> WorkflowState:
    return state

def generate_post_node(state: WorkflowState) -> WorkflowState:
    return state

def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("get_input", get_input_node)
    graph.add_node("analyze_prompt", analyze_prompt_node)
    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("generate_post", generate_post_node)

    graph.set_entry_point("get_input")
    graph.add_edge("get_input", "analyze_prompt")
    graph.add_edge("analyze_prompt", "hybrid_search")
    graph.add_edge("hybrid_search", "generate_post")
    graph.add_edge("generate_post", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_graph()

    initial_state: WorkflowState = {
        "user_prompt": "",
        "search_query": None,
        "keywords": [],
        "is_valid": False,
        "invalid_reason": None,
        "search_results": [],
    }

    final_state = app.invoke(initial_state)