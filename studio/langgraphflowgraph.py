# File: langgraphflowgraph.py (place in the root dynabot directory)

"""
LangGraph Studio Compatible Version
This version mocks external dependencies to allow graph visualization in LangGraph Studio
"""

import os
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    query: str
    original_query: str
    chat_history: str 
    db_name: str
    selected_file_names: List[str]
    documents: List[dict] 
    answer: str
    relevance_score: int
    retry_count: int
    search_kwargs: dict
    search_index_name: str
    initial_answer: str
    user_decision: str
    research_mode: bool
    external_context: Annotated[list, operator.add] 
    generated_search_query: str 


# --- NODE FUNCTIONS ---

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieves documents from MongoDB vector store"""
    print("---RETRIEVING DOCUMENTS---")
    state["documents"] = [
        {
            "page_content": "Sample document content",
            "metadata": {"file_name": "sample.pdf"}
        }
    ]
    return state


def optimize_query_node(state: GraphState) -> dict:
    """Generates optimized search query based on retrieved documents"""
    print("---OPTIMIZING SEARCH QUERY---")
    query = state.get("query", "")
    documents = state.get("documents", [])
    
    # Extract filenames
    unique_files = set()
    for doc in documents:
        meta = doc.get("metadata", {})
        actual_name = meta.get("file_name")
        if actual_name:
            unique_files.add(actual_name)
    
    context_str = ", ".join(unique_files)
    
    if not context_str:
        return {"generated_search_query": query}
    
    # Simple mock optimization
    optimized = f"{query} {context_str}"
    print(f"--- Query Optimized: '{query}' -> '{optimized}' ---")
    return {"generated_search_query": optimized}


def search_web(state: GraphState) -> dict:
    """Searches the general web via Tavily"""
    print("---EXECUTING WEB SEARCH---")
    query = state.get("generated_search_query", state.get("query"))
    tavily_text = f"- Web result 1 about {query}\n- Web result 2 about {query}"
    return {"external_context": [f"### WEB SEARCH RESULTS:\n{tavily_text}"]}


def search_wiki(state: GraphState) -> dict:
    """Searches Wikipedia"""
    print("---EXECUTING WIKI SEARCH---")
    query = state.get("generated_search_query", state.get("query"))
    wiki_text = f"- Wikipedia article about {query}..."
    return {"external_context": [f"### WIKIPEDIA RESULTS:\n{wiki_text}"]}


def search_arxiv(state: GraphState) -> dict:
    """Searches Academic Papers"""
    print("---EXECUTING ARXIV SEARCH---")
    query = state.get("generated_search_query", state.get("query"))
    arxiv_text = f"- Title: Research Paper on {query}\n  Abstract: This paper discusses..."
    return {"external_context": [f"### ARXIV PAPERS:\n{arxiv_text}"]}


def generate_answer(state: GraphState) -> GraphState:
    """Generates answer from internal and external context"""
    print("---GENERATING ANSWER---")
    query = state.get("query", "")
    documents = state.get("documents", [])
    external_list = state.get("external_context", [])
    
    internal_context = "\n\n".join([doc["page_content"] for doc in documents])
    external_text = "\n\n".join(external_list)
    
    if external_text:
        combined_context = f"=== INTERNAL ===\n{internal_context}\n\n=== EXTERNAL ===\n{external_text}"
    else:
        combined_context = internal_context
    
    answer = f"Based on the documents and research, here's an answer to '{query}'..."
    state["answer"] = answer
    return state


def evaluate_answer(state: GraphState) -> dict:
    """Evaluates answer relevance"""
    print("---EVALUATING ANSWER---")
    # Mock evaluation score
    score = 7  # Mock score between 1-10
    state["relevance_score"] = score
    return state


def retry_counter(state: GraphState) -> GraphState:
    """Increments retry count"""
    retry_count = state.get("retry_count", 0)
    state["retry_count"] = retry_count + 1
    print(f"Retry count: {state['retry_count']}")
    return state


def generate_better_prompt(state: GraphState) -> GraphState:
    """Rewrites query for better retrieval"""
    print("---REWRITING QUERY---")
    original_query = state.get("original_query", "")
    new_query = f"Refined: {original_query}"
    state["query"] = new_query
    state["answer"] = ""
    state["documents"] = []
    print(f"NEW QUERY: {new_query}")
    return state


def expand_retrieval(state: GraphState) -> GraphState:
    """Expands search by increasing k value"""
    current_k = state.get("search_kwargs", {}).get("k", 5)
    new_k = current_k + 5
    state["search_kwargs"] = {"k": new_k}
    state["answer"] = ""
    state["documents"] = []
    print(f"Expanding retrieval to k={new_k}")
    return state


def handle_failure(state: GraphState) -> GraphState:
    """Handles failure case"""
    state["answer"] = "I apologize, but I was unable to find a relevant answer..."
    print("Bad answer - handling failure")
    return state


def pass_answer(state: GraphState) -> GraphState:
    """Passes answer through"""
    print("Answer passed - flow complete")
    return state


# --- ROUTING FUNCTIONS ---

def route_research(state: GraphState) -> str:
    """Routes to research mode or direct generation"""
    if state.get("research_mode"):
        return "optimize_query"
    else:
        return "generate"


def route_approval(state: GraphState) -> str:
    """Routes based on user approval decision"""
    return state.get("user_decision", "retry")


# --- GRAPH BUILDER ---

def build_graph() -> StateGraph:
    """Builds and compiles the RAG workflow graph"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("search_web", search_web)
    workflow.add_node("search_wiki", search_wiki)
    workflow.add_node("search_arxiv", search_arxiv)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate", evaluate_answer)
    workflow.add_node("retry_counter", retry_counter)
    workflow.add_node("rewrite_query", generate_better_prompt)
    workflow.add_node("expand_retrieval", expand_retrieval)
    workflow.add_node("handle_failure", handle_failure)
    workflow.add_node("pass_answer", pass_answer)
    workflow.add_node("human_approval", lambda state: state)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add conditional edge: research mode routing
    workflow.add_conditional_edges(
        "retrieve",
        route_research,
        {
            "optimize_query": "optimize_query",
            "generate": "generate"
        }
    )
    
    # Add parallel edges from optimize_query
    workflow.add_edge("optimize_query", "search_web")
    workflow.add_edge("optimize_query", "search_wiki")
    workflow.add_edge("optimize_query", "search_arxiv")
    # Merge parallel searches to generate
    workflow.add_edge("search_web", "generate")
    workflow.add_edge("search_wiki", "generate")
    workflow.add_edge("search_arxiv", "generate")
    
    # Generation flow
    workflow.add_edge("generate", "evaluate")
    
    # Evaluation routing
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: "pass" if state["relevance_score"] > 5 else "fail",
        {
            "pass": "pass_answer",
            "fail": "retry_counter"
        }
    )
    
    # Retry routing
    workflow.add_conditional_edges(
        "retry_counter",
        lambda state: state["retry_count"],
        {
            1: "rewrite_query",
            2: "expand_retrieval",
            3: "handle_failure"
        }
    )
    
    # Human approval flow
    workflow.add_edge("rewrite_query", "human_approval")
    workflow.add_conditional_edges(
        "human_approval",
        route_approval,
        {
            "retry": "retrieve",
            "expand": "expand_retrieval"
        }
    )
    
    # Loop back
    workflow.add_edge("expand_retrieval", "retrieve")
    
    # End states
    workflow.add_edge("pass_answer", END)
    workflow.add_edge("handle_failure", END)
    
    return workflow.compile()


# Export for LangGraph Studio
graph = build_graph()