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

from db_utils import get_collection, set_embedding_model
import streamlit as st


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
        

def generate_answer(state: GraphState) -> GraphState:
    query = state.get("query", "")
    
    chat_history = state.get("chat_history", "") 
    documents = state.get("documents", [])
    
    # Access via dictionary key
    context = "\n\n".join([doc["page_content"] for doc in documents])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are a chatbot answering questions about documents uploaded by a user. Use the provided context to answer the question.
        You may infer reasonable conclusions if they logically follow from the context and extend the answer.
        Do not mention the existence of "context" or "document" in your response. Do not mention anything referring to the document provided.
        The context you see has already been collected from the documents for you to answer questions from. 
        
        You have been provided the conversation history in "Previous Conversation History" for a limited number of the previous messages which you can use to
        infer things about the context of the user's question (e.g., what pronouns like "it" or "they" refer to). You can make logical deductions.

        
        Make your answer readable by utilizing bullet points whenever possible, but do not unnecessarily use them in cases
        where text formatted in paragraphs would be more readable. Do not give extremely long or short answers unless the user asks specifically 
        or unless the genuine content of the answer is brief or extremely lengthy itself, in which you may resort to giving extremely long or short
        answers. Keep general answers to a medium length.

        **Important:** If the provided Context contains interesting related details that were not specifically asked for, 
        you are highly encouraged to end your response by asking: "Would you like to know more about [Related Topic]?"

    
        Previous Conversation history:
        {chat_history}


        Context: {context}


        Question: {query}
        Answer:"""
    )
    
    rag_chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    
    try:
        # Pass chat_history to the chain
        answer = rag_chain.invoke({
            "query": query, 
            "context": context, 
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
      """You are an expert answer evaluator for a RAG system. Your task is to determine the relevance and accuracy of a generated answer by an llm
        to a user's query relative to the and based on the yser query and the provided documents that act as context which has to be used 
        as a primary source to generate the answer. 

    Criteria for grading:
    1. Groundedness: The answer must be based primarily on the provided Retrieved Documents. It can contain logical deductions and common sense but not hallucinations.
    2. Relevance: The answer must directly address the User Query. (MOST IMPORTANT)
    3. Completeness: The answer should utilize the context fully to provide a helpful response.

    Score the answer on a scale from 1 to 10:
    - 1: The answer is hallucinated, factually incorrect according to the docs, or completely irrelevant.
    - 5: The answer touches on the topic but misses key details or includes some unsupported claims.
    - 10: The answer is perfectly grounded in the documents and fully answers the user's question.

    Respond with ONLY the numerical score (e.g., 8).
        
        User Query: {query}
        Retrieved Documents: {documents}
        Generated Answer: {answer}

        Relevance Score (1-10):"""
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
      """You are a query re-writer. The user asked a query, relevant context information found in a pre-decided database
      was fetched, and an answer was generated by an LLM. The initial retrieval for the user's query was unsuccessful.
        To help you, here is the original query, the previous conversation history, and the content from the context that was retrieved initially. 
        
        Your task:
        1. Look at the "Original Query" and "Previous Conversation History" to understand what the user is truly asking (resolve pronouns like 'it', 'he', 'that').
        2. Analyze the "Initially Retrieved Content" to see why it failed.
        3. Rephrase the query to be specific and optimized for vector similarity search.
        
        Only return the rephrased query without any additional text.

        Original Query: {original_query}

        Previous Conversation History:
        {chat_history}

        Initially Retrieved Content:
        {retrieved_content}

        Rephrased Query:"""
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

def build_rag_graph():
    memory = MemorySaver()
 
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate", evaluate_answer)
    workflow.add_node("retry_counter", retry_counter)
    workflow.add_node("rewrite_query", generate_better_prompt)
    workflow.add_node("expand_retrieval", expand_retrieval)
    workflow.add_node("handle_failure", handle_failure)
    workflow.add_node("pass_answer", pass_answer)
    
    workflow.add_node("human_approval", lambda state: state)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: "pass" if state["relevance_score"] > 5 else "fail",
        {
            "pass": "pass_answer",
            "fail": "retry_counter"
        }
    )
    
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
            "retry": "retrieve",
            "expand": "expand_retrieval"
        }
    )

    workflow.add_edge("expand_retrieval", "retrieve")
    workflow.add_edge("pass_answer", END)
    workflow.add_edge("handle_failure", END)

    app = workflow.compile(
        checkpointer=memory, 
        interrupt_before=["human_approval"]
    )
    print("LangGraph RAG workflow with self-correction compiled successfully.")
    return app