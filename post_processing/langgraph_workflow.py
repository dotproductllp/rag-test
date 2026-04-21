from typing import TypedDict, List, Optional, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from llm_search_query import QueryGenerator, INVALID_QUERY_MARKER
from hybrid_search import HybridSearch
from generate_post import PostGenerator

load_dotenv()

query_generator = QueryGenerator()
hybrid_search = HybridSearch()
post_generator = PostGenerator()

class SearchResult(TypedDict):
    id: str
    score: float
    article_body: str

class WorkflowState(TypedDict):
    user_prompt: str
    search_query: Optional[str]
    keywords: List[str]
    is_valid: bool
    invalid_reason: Optional[str]
    search_results: List[SearchResult]
    generated_post: Optional[str]

def get_input_node(state: WorkflowState) -> WorkflowState:
    """Fetch prompt from the user. On retry show the reason."""

    if state.get("invalid_reason"):
        print(f"\n! Prompt rejected: {state['invalid_reason']}")

    user_prompt = input("\nEnter your prompt (or type 'exit' to quit): ").strip()

    if user_prompt.lower() == "exit":
        return {
            **state,
            "user_prompt": "exit",
            "is_valid": True,
            "invalid_reason": None,
        }
    else:
        return {
            **state,
            "user_prompt": user_prompt,
            "invalid_reason": None,
        }

MIN_PROMPT_CHARS = 20

def analyze_prompt_node(state: WorkflowState) -> WorkflowState:
    """
    length check, then call LLM that both generates the
    search query, keywords and flags 'invalid query'.
    """

    prompt = state["user_prompt"]
    if len(prompt) < MIN_PROMPT_CHARS:
        return {
            **state,
            "is_valid": False,
            "invalid_reason": (
                f"Prompt is too short. Only ({len(prompt)} chars). Enter at least {MIN_PROMPT_CHARS} chars."
            ),
            "search_query": None,
            "keywords": [],
        }

    query = query_generator.generate_query_keyword(prompt)

    if query["search_query"].strip().lower() == INVALID_QUERY_MARKER:
        return {
            **state,
            "is_valid": False,
            "invalid_reason": query.get("invalid_reason"),
            "search_query": None,
            "keywords": [],
        }
    else:
        return {
            **state,
            "is_valid": True,
            "invalid_reason": None,
            "search_query": query["search_query"],
            "keywords": query["keywords"],
        }

def hybrid_search_node(state: WorkflowState) -> WorkflowState:
    """hybrid search and put (id, score, body) into state."""

    results = hybrid_search.perform_hybrid_search(
        semantic_query=state["search_query"],
        required_keywords=state["keywords"],
        top_k=10,
    )

    extracted_posts: List[SearchResult] = [
        {
            "id": r["id"],
            "score": float(r.get("score")),
            "article_body": r.get("article_body"),
        }
        for r in results
    ]

    return {**state, "search_results": extracted_posts}

def generate_post_node(state: WorkflowState) -> WorkflowState:
    """Passes user prompt and search results to the LLM to generate the final post."""

    generated_content = post_generator.generate_post(
        user_prompt=state["user_prompt"],
        search_results=state["search_results"]
    )
    print(generated_content)
    return {**state, "generated_post": generated_content}


def route_after_analyze(state: WorkflowState) -> Literal["get_input", "hybrid_search", "__end__"]:
    """If the user typed exit, end. If invalid, loop back. Otherwise move on."""

    if state.get("user_prompt").lower() == "exit":
        return "__end__"
    if not state.get("is_valid", False):
        return "get_input"
    return "hybrid_search"


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("get_input", get_input_node)
    graph.add_node("analyze_prompt", analyze_prompt_node)
    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("generate_post", generate_post_node)

    graph.set_entry_point("get_input")
    graph.add_edge("get_input", "analyze_prompt")
    graph.add_conditional_edges(
        "analyze_prompt",
        route_after_analyze,
        {
            "get_input": "get_input",
            "hybrid_search": "hybrid_search",
            "__end__": END,
        },
    )
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
        "generated_post": None,
    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 100})

    if final_state.get("user_prompt").lower() == "exit":
        print("Exit")
    else:
        print("\nWorkflow complete")
