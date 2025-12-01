import os
from typing import TypedDict, List, Any
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents import Document 

from utils.db_utils import get_collection, set_embedding_model
from extras.prompts import GENERATE_ANSWER,EVALUATE_ANSWER,REWRITE_ANSWER
import streamlit as st
try:
   
    from pydantic import BaseModel, Field
except Exception:
    
    from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated
import operator
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


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

def get_retriever(search_index_name: str, file_names_filter: List[str], k:int=5):
    embeddings = set_embedding_model()
    collection = get_collection()

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=search_index_name,
        relevance_score_fn="cosine",
        embedding_key="vector_embedding"
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",      
        search_kwargs={"k": k}          
    )

    return retriever

def retrieve_documents(state: GraphState) -> GraphState:
 ''' #debugging code
    st.sidebar.markdown("---")
    st.sidebar.subheader("Node Debugger")
    
    r_mode = state.get('research_mode')
    st.sidebar.write(f"**Current Node:** retrieve_documents")
    st.sidebar.write(f"**Research Mode:** `{r_mode}`")
    st.sidebar.write(f"**Query:** {state.get('query')}")'''

    query = state.get("query", "")
    selected_file_names = state.get("selected_file_names", [])
    search_index_name = state.get("search_index_name", "")

    k_value = state.get("search_kwargs", {}).get("k", 5)

    try:
        retriever = get_retriever(search_index_name, selected_file_names, k=k_value)
        mongo_filter = {"metadata.file_name": {"$in": selected_file_names}}
        
        # Invoke retriever
        raw_documents = retriever.invoke(query, config={"search_kwargs": {"pre_filter": mongo_filter}})
        
        #  Sanitize Metadata (Convert ObjectId to string) 
        serialized_docs = []
        for doc in raw_documents:
            # Copy metadata to allow modification
            clean_metadata = doc.metadata.copy()
            
            # Iterate over keys to find ObjectId or other non-serializable types
            for key, value in clean_metadata.items():
                # If the value is not a standard primitive, convert to string
                # This catches ObjectId, Dates, etc.
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    clean_metadata[key] = str(value)
            
            serialized_docs.append({
                "page_content": doc.page_content,
                "metadata": clean_metadata
            })
        
        
        state["documents"] = serialized_docs
        
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        state["documents"] = []
    
    st.toast("Retrieved documents for query")
    return state
    

def optimize_query_node(state: GraphState):
    """
    Generates a specific search query based on the retrieved documents.
    Runs ONCE before the parallel search.
    """
    print("---OPTIMIZING SEARCH QUERY---")
    query = state.get("query", "")
    documents = state.get("documents", [])
    
    # 1. Extract Filenames
    unique_files = set()
    for doc in documents:
        meta = doc.get("metadata", {})
        actual_name = meta.get("file_name")
        if not actual_name:
            raw_source = meta.get("source", "")
            if raw_source:
                actual_name = os.path.basename(raw_source)
        if actual_name:
            unique_files.add(actual_name)

    context_str = ", ".join(unique_files)
    
    # If no docs found, just use original query
    if not context_str:
        return {"generated_search_query": query}

    # 2. LLM Call
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    prompt = f"""
    User Query: {query}
    Context: The user is asking about these specific files: {context_str}.
    
    Task: Write a specific web search query that combines the User Query with the File Context. 
    Example: Query: "revenue" + Files: "Nike_2023.pdf" -> "Nike 2023 revenue"
    
    Return ONLY the new query string.
    """
    
    try:
        new_query = llm.invoke(prompt).content.strip()
        print(f"--- Query Optimized: '{query}' -> '{new_query}' ---")
        return {"generated_search_query": new_query}
    except:
        return {"generated_search_query": query}

def search_web(state: GraphState):
    """Searches the general web via Tavily"""
    print("---EXECUTING WEB SEARCH---")
    query = state.get("generated_search_query", state.get("query"))
    try:
        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke({"query": query})
        
       
        tavily_text = "\n".join([f"- {res['content']} (Source: {res['url']})" for res in results])
        return {"external_context": [f"### WEB SEARCH RESULTS:\n{tavily_text}"]}
    except Exception as e:
        print(f"Tavily Error: {e}")
        return {"external_context": []} 

def search_wiki(state: GraphState):
    query = state.get("generated_search_query", state.get("query"))
    try:
        
        tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000))
        
        
        result = tool.invoke({"query": query})
        
        return {"external_context": [f"### WIKIPEDIA RESULTS:\n{result}"]}
    except Exception as e:
        print(f"Wiki Error: {e}")
        return {"external_context": []}


def search_arxiv(state: GraphState):
    query = state.get("generated_search_query", state.get("query"))
    try:
        
        tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000))
        
        
        result = tool.invoke({"query": query})
        
        return {"external_context": [f"### ARXIV PAPERS:\n{result}"]}
    except Exception as e:
        print(f"Arxiv Error: {e}")
        return {"external_context": []}

def generate_answer(state: GraphState) -> GraphState:
    query = state.get("query", "")
    chat_history = state.get("chat_history", "") 
    documents = state.get("documents", [])
    
    
    internal_context = "\n\n".join([doc["page_content"] for doc in documents])
    
   
    external_list = state.get("external_context", [])
    external_text = "\n\n".join(external_list)
    
   
    if external_text:
        combined_context = f"""
        === INTERNAL DOCUMENTS ===
        {internal_context}
        
        === EXTERNAL RESEARCH ===
        {external_text}
        """
    else:
        combined_context = internal_context
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_template(
       GENERATE_ANSWER
    )
    
    rag_chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    
    try:
        # Pass combined_context to the chain as 'context'
        answer = rag_chain.invoke({
            "query": query, 
            "context": combined_context, 
            "chat_history": chat_history
        })
        state["answer"] = answer
    
       
    except Exception as e:
        print(f"Error during answer generation: {e}")
        state["answer"] = "I apologize, but I encountered an error while generating the answer."
    
    st.toast("Generated answer for query")
 
    return state


def evaluate_answer(state: GraphState) -> dict:
    query = state.get("query", "")
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    evaluation_prompt = ChatPromptTemplate.from_template(
     EVALUATE_ANSWER
    )
    
    evaluator = (evaluation_prompt | llm | StrOutputParser())
    
    try:
        if not documents:
            score=1
        else:
            raw_score = evaluator.invoke({"query": query, "documents": documents, "answer": answer}).strip()
            try:
                score = int(raw_score)
                score = max(1, min(10,score)) 
            except ValueError:
                score = 1
    except Exception as e:
        print(f"Error during answer evaluation: {e}")
        score=1

    print(f"Answer relevance score: {score}")
    state["relevance_score"] = score
    st.toast("Evaluated answer relevance")
    return state

def retry_counter(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0)
    state["retry_count"] = retry_count + 1
    st.toast("retry count increased")
    return state

def generate_better_prompt(state: GraphState) -> GraphState:
    query = state.get("query", "")
    
    chat_history = state.get("chat_history", "")
    documents = state.get("documents", [])
    
   
    retrieved_content = "\n\n".join([doc["page_content"] for doc in documents])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    
  
    reprompt_template = ChatPromptTemplate.from_template(
     REWRITE_ANSWER
    )
    
    rephraser = (reprompt_template | llm | StrOutputParser())
    
    try:
        
        new_query = rephraser.invoke({
            "original_query": query,
            "retrieved_content": retrieved_content,
            "chat_history": chat_history
        }).strip()
        
        state["query"] = new_query
        state["answer"] = "" 
        state["documents"] = []  
        
        print(f"NEW QUERY: {new_query}")
        st.toast(f"Retrying with: {new_query[:50]}...")
        
    except Exception as e:
        print(f"Error during prompt generation: {e}")
        st.toast("Failed to rewrite query, keeping original")
    
    return state

def expand_retrieval(state: GraphState) -> GraphState:
    current_k = state.get("search_kwargs", {}).get("k", 5)
    new_k = current_k + 5 
    
    state["search_kwargs"] = {"k": new_k}
    state["answer"] = "" 
    state["documents"] = []  
    
    print(f"Expanding retrieval to k={new_k}")
    st.toast(f"Expanding search to {new_k} documents")
    return state

def handle_failure(state: GraphState) -> GraphState:
    state["answer"] = "I apologize, but I was unable to find a relevant answer in the uploaded file(s). Please try rephrasing your question for better results. "
    st.toast("bad answer.")
    return state
    

def pass_answer(state: GraphState) -> GraphState:
    st.toast("answer passed")
    return state

def route_approval(state: GraphState) -> str:
    return state.get("user_decision", "retry")



def route_research(state: GraphState):
    """
    Router: Checks if Research Mode is toggled ON.
    """

    mode = state.get("research_mode", False)
    
    print(f"\n[ROUTER CHECK] Research Mode: {mode}")
    print(f"[ROUTER CHECK] State Keys Present: {list(state.keys())}\n")
    
    if mode:
        return "optimize_query" 
    else:
        return "generate"

def build_rag_graph():
    memory = MemorySaver()
    workflow = StateGraph(GraphState)

    
    # Core RAG Nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate", evaluate_answer)
    
    # Research Nodes (New)
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("search_web", search_web)
    workflow.add_node("search_wiki", search_wiki)
    workflow.add_node("search_arxiv", search_arxiv)
    
    # Loop/Logic Nodes
    workflow.add_node("retry_counter", retry_counter)
    workflow.add_node("rewrite_query", generate_better_prompt)
    workflow.add_node("expand_retrieval", expand_retrieval)
    workflow.add_node("handle_failure", handle_failure)
    workflow.add_node("pass_answer", pass_answer)
    workflow.add_node("human_approval", lambda state: state) # Dummy node for pause
    

    
    # Start Point
    workflow.set_entry_point("retrieve")
    
  
    # After retrieving internal docs, check if we need external research
    workflow.add_conditional_edges(
        "retrieve",
        route_research, 
        {
            "optimize_query": "optimize_query", # Go to research optimization
            "generate": "generate"              # Skip to generation
        }
    )
    

    
    workflow.add_edge("optimize_query", "search_web")
    workflow.add_edge("optimize_query", "search_wiki")
    workflow.add_edge("optimize_query", "search_arxiv")
    
    
    workflow.add_edge("search_web", "generate")
    workflow.add_edge("search_wiki", "generate")
    workflow.add_edge("search_arxiv", "generate")
    
    
    workflow.add_edge("generate", "evaluate")
    
    # Check Score
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: "pass" if state["relevance_score"] > 5 else "fail",
        {
            "pass": "pass_answer",
            "fail": "retry_counter"
        }
    )
    
    # Retry Logic
    workflow.add_conditional_edges(
        "retry_counter",
        lambda state: state["retry_count"], 
        {
            1: "rewrite_query",
            2: "expand_retrieval",
            3: "handle_failure"
        }
    )

  
    workflow.add_edge("rewrite_query", "human_approval")
    
    workflow.add_conditional_edges(
        "human_approval",
        route_approval,
        {
            "retry": "retrieve",          # Restart with new query
            "expand": "expand_retrieval"  # Restart with broader search
        }
    )

   
    workflow.add_edge("expand_retrieval", "retrieve")
    workflow.add_edge("pass_answer", END)
    workflow.add_edge("handle_failure", END)


    app = workflow.compile(
        checkpointer=memory, 
        interrupt_before=["human_approval"]
    )
    
    print("LangGraph RAG workflow compiled successfully.")
    return app