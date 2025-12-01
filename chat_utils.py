import streamlit as st
from langgraph_flow import build_rag_graph

@st.cache_resource(show_spinner=False)
def get_rag_graph():
    return build_rag_graph()


rag_graph_app = get_rag_graph()


def get_formatted_history(chat_key):
    """
    Formats history string based on message count rules:
    - If total history < 5 messages: Keep last 1 pair (User/AI).
    - If total history >= 5 messages: Keep last 2 pairs (User/AI/User/AI).
    """
    if chat_key not in st.session_state.chat_history:
        return ""
    
    # Get all messages EXCEPT the current one
    full_history = st.session_state.chat_history[chat_key][:-1]
    
    total_messages = len(full_history)
    
    if total_messages == 0:
        return ""
        
    limit = 2 if total_messages < 5 else 4
    subset_history = full_history[-limit:]
    
    history_str = ""
    for msg in subset_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_str += f"{role}: {content}\n"
        
    return history_str


def run_rag_graph(input_state=None, chat_key=None):
    """
    Runs the graph. If input_state is None, it resumes from the checkpoint.
    """
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.spinner("Processing..."):
        # Run the graph
        for event in rag_graph_app.stream(input_state, config=config, stream_mode="values"):
            pass 
        
        # Check execution status
        snapshot = rag_graph_app.get_state(config)

        if snapshot.next and "human_approval" in snapshot.next:
            st.session_state.waiting_for_approval = True
            st.rerun()

        else:
            final_answer = snapshot.values.get("answer", "No answer generated.")
            st.session_state.chat_history[chat_key].append({"role": "assistant", "content": final_answer})
            st.session_state.waiting_for_approval = False