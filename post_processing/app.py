import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph_workflow import WorkflowState, analyze_prompt_node, hybrid_search_node, generate_post_node

@st.cache_resource
def ui_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("analyze_prompt", analyze_prompt_node)
    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("generate_post", generate_post_node)

    graph.set_entry_point("analyze_prompt")

    def route_after_analyze(state: WorkflowState):
        if not state.get("is_valid", False):
            return "__end__"
        return "hybrid_search"
        
    graph.add_conditional_edges(
        "analyze_prompt",
        route_after_analyze,
        {
            "hybrid_search": "hybrid_search",
            "__end__": END,
        },
    )
    graph.add_edge("hybrid_search", "generate_post")
    graph.add_edge("generate_post", END)
    
    return graph.compile()

app = ui_graph()

st.set_page_config(page_title="LinkedIn Post Generator", layout="centered")
st.title("LinkedIn Post Generator")

with st.form("prompt_form"):
    user_prompt = st.text_area(
        "Enter a prompt:",
        height=50,
    )
    submitted = st.form_submit_button("Generate Post")

if submitted:
    if not user_prompt.strip():
        st.warning("Please enter a prompt before submitting.")
    else:
        with st.spinner("Analyzing prompt"):
            initial_state: WorkflowState = {
                "user_prompt": user_prompt,
                "search_query": None,
                "keywords": [],
                "is_valid": False,
                "invalid_reason": None,
                "search_results": [],
                "generated_post": None,
            }
            final_state = app.invoke(initial_state)

        if not final_state.get("is_valid"):
            st.error(f"**Prompt Rejected:** {final_state.get('invalid_reason')}")
        else:
            with st.expander("🔍 View AI Search Details & Scraped Context"):
                for i, res in enumerate(final_state["search_results"]):
                    st.markdown(f"**Post {i+1}** (Score: `{res['score']:.4f}`)\n> {res['article_body'][:200]}...")

            st.success("Post Generated Successfully!")
            st.markdown("### Your LinkedIn Post:")
            st.write(final_state["generated_post"])
