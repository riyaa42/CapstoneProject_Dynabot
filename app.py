from dotenv import load_dotenv 
import streamlit as st

if "selected_file_name" not in st.session_state: 
    st.session_state.selected_file_name=None
    
import tempfile
from streamlit_pdf_viewer import pdf_viewer
from streamlit_extras.stylable_container import stylable_container
import os
load_dotenv()
from data_processing import load_file, split_docs, convert_pptx_to_pdf
from db_utils import check_env, cleanup, add_documents, delete_file,search_index
from langgraph_flow import build_rag_graph, GraphState
import uuid


@st.cache_resource(show_spinner=False)
def get_rag_graph():
    return build_rag_graph()

rag_graph_app = get_rag_graph()

st.set_page_config(layout="wide", page_title="DynaBOT")

st.markdown("""
<style>
    /* Main app background - black */
    .stApp {
        background-color: #000000;
    .stAppHeader {
        background-color: #000000;}
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }


    /* Multiselect input wrapper - pink border */
    div[data-testid="stMultiSelect"] > div {
        border-color: #F25081 !important;
    }

    /* Multiselect inner div - pink border */
    div[data-testid="stMultiSelect"] > div > div {
        border-color: #F25081 !important;
        background-color: #1a1a1a !important;
    }

    /* Multiselect actual input component - pink border */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] {
        border-color: #F25081 !important;
        background-color: #1a1a1a !important;
    }

    /* Selected tags/pills - pink background, white text */
    span[data-baseweb="tag"] {
        background-color: #F25081 !important;
        color: #ffffff !important;
    }

    /* Dropdown menu container - dark background */
    div[role="listbox"] {
        background-color: #1a1a1a !important;
        border-color: #F25081 !important;
    }

    /* Dropdown menu items - white text on dark */
    div[role="listbox"] li {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Dropdown menu selected items - pink background */
    div[role="listbox"] li[aria-selected="true"] {
        background-color: #F25081 !important;
        color: #ffffff !important;
    }

    /* Placeholder text - gray */
    div[data-testid="stMultiSelect"] input::placeholder {
        color: #888888 !important;
    }
    .stSpinner > div > div {
        border-top-color: #F25081;
        color:#F25081 !important;

    .stChatInputContainer > div > div > input {
        border-color: #F25081 !important;
        background-color: #1a1a1a;
        color: #ffffff !important;
    }
    .stChatInputContainer > div > div > input:focus {
        border-color: #F25081 !important;
        box-shadow: 0 0 0 0.2rem rgba(242, 80, 129, 0.25) !important;
    }

   
}

</style>
""", unsafe_allow_html=True)

    

try:
    check_env()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

#testing new feature start
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4()) 

if "waiting_for_approval" not in st.session_state:
    st.session_state.waiting_for_approval = False
#testing new feature end

if "processed_file_info" not in st.session_state:
    st.session_state.processed_file_info={} # key:file_name {str} ,value:{ tmp_path=str, ingestion=bool}


if "chat_history" not in st.session_state:
    st.session_state.chat_history = {} #key: file_name {str}, value: prev chat messages{List}

if "shown_toasts" not in st.session_state:
    st.session_state.shown_toasts=[]

if "orphan_cleanup" not in st.session_state:
    st.session_state.orphan_cleanup = False

if not st.session_state.orphan_cleanup:
    try:
        store_db = cleanup()
        uploaded_files_names_set = {f.name for f in st.session_state.uploaded_files} if "uploaded_files" in st.session_state else set()
        orphan_files=store_db-uploaded_files_names_set

        if orphan_files:
            for file_name in orphan_files:
                delete_file(file_name)
        st.session_state.orphan_cleanup = True
    except Exception as e:
        st.toast(f"Error during initial DB cleanup: {e}")
        st.session_state.initial_db_cleanup_done = True


def run_rag_graph(input_state=None, chat_key=None):
    """
    Runs the graph. If input_state is None, it resumes from the checkpoint.
    """
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.spinner("Processing..."):
        # Run the graph (stream_mode="values" returns the full state at each step)
        # If input_state is None, LangGraph resumes automatically using thread_id
        for event in rag_graph_app.stream(input_state, config=config, stream_mode="values"):
            pass 
        
        # Check execution status
        snapshot = rag_graph_app.get_state(config)
        
        # A. Did we stop at the 'human_approval' node?
        if snapshot.next and "human_approval" in snapshot.next:
            st.session_state.waiting_for_approval = True
            st.rerun() # Rerun to hide chat_input and show form
            
        # B. Did we finish successfully?
        else:
            final_answer = snapshot.values.get("answer", "No answer generated.")
            st.session_state.chat_history[chat_key].append({"role": "assistant", "content": final_answer})
            st.session_state.waiting_for_approval = False
            # We do NOT rerun here, just let the script finish to show the answer

#testing new feature

def get_formatted_history(chat_key):
    """
    Formats history string based on message count rules:
    - If total history < 5 messages: Keep last 1 pair (User/AI).
    - If total history >= 5 messages: Keep last 2 pairs (User/AI/User/AI).
    """
    if chat_key not in st.session_state.chat_history:
        return ""
    
    # Get all messages EXCEPT the current one (the user just typed it)
    # We want history, not the current query repeated
    full_history = st.session_state.chat_history[chat_key][:-1]
    
    total_messages = len(full_history)
    
    if total_messages == 0:
        return ""
        
    # LOGIC: 
    # If messages < 5, keep last 2 messages (1 interaction pair)
    # If messages >= 5, keep last 4 messages (2 interaction pairs)
    limit = 2 if total_messages < 5 else 4
    
    # Slice the list
    subset_history = full_history[-limit:]
    
    # Format as string
    history_str = ""
    for msg in subset_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_str += f"{role}: {content}\n"
        
    return history_str


uploaded_files=st.sidebar.file_uploader("**Upload a file**", type=["pdf","pptx"], accept_multiple_files=True)
selected_file_names = []


for uploaded_file in uploaded_files:
    if uploaded_file.name not in st.session_state.shown_toasts:
        st.toast(f"Uploaded file: {uploaded_file.name}", icon="🟣")
        st.session_state.shown_toasts.append(uploaded_file.name)

if uploaded_files:
    uploaded_files_names = {f.name for f in uploaded_files}
else:
    uploaded_files_names = set()

newly_added_files=[]
for uploaded_file in uploaded_files:
    if uploaded_file.name not in st.session_state.processed_file_info:
        newly_added_files.append(uploaded_file)
        

if newly_added_files:
    for uploaded_file in newly_added_files:
        file_name=uploaded_file.name
        tmp_path=None
        pdf_viewer_path=None

        try:
            file_ext = os.path.splitext(file_name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            
            with st.spinner(f"Processing {file_name}..."):
                if file_ext == ".pptx":
                     pdf_viewer_path = convert_pptx_to_pdf(tmp_path, os.path.dirname(tmp_path))
                     if not pdf_viewer_path:
                        st.error(f"Failed to convert {file_name} to PDF.")
                     else:
                        tmp_path = pdf_viewer_path
                elif file_ext == ".pdf":
                    pdf_viewer_path = tmp_path  


                docs=load_file(tmp_path)
                chunks=split_docs(docs)
                
                add_documents(chunks,file_name)
                st.toast("added documents")

                st.session_state.processed_file_info[file_name] = {
                    "tmp_path": tmp_path,
                    "pdf_viewer_path": pdf_viewer_path,
                    "ingested": True
                }

                if file_name not in st.session_state.chat_history:
                    st.session_state.chat_history[file_name] = []

        except Exception as e:

                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

                delete_file(file_name)
                if file_name in st.session_state.processed_file_info:
                    del st.session_state.processed_file_info[file_name]
                if file_name in st.session_state.chat_history:
                    del st.session_state.chat_history[file_name]
                st.toast("file not ingested")    

files_to_remove=[]
for file_name in st.session_state.processed_file_info.keys():
    if file_name not in uploaded_files_names:
        files_to_remove.append(file_name)

if files_to_remove:
    for file_name in files_to_remove:
        key_value=st.session_state.processed_file_info[file_name]
        tmp_path=key_value["tmp_path"]
        pdf_viewer_path = key_value.get("pdf_viewer_path")

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if pdf_viewer_path and os.path.exists(pdf_viewer_path) and pdf_viewer_path != tmp_path:
            os.remove(pdf_viewer_path)
        delete_file(file_name)
        del st.session_state.processed_file_info[file_name]
        if file_name in st.session_state.chat_history:
            del st.session_state.chat_history[file_name]

        if st.session_state.selected_file_name==file_name:
            st.session_state.selected_file_name=None

st.sidebar.markdown("**Select a File**")
available_files=list(st.session_state.processed_file_info.keys())

if not available_files:
     st.sidebar.markdown("<span style='color:deeppink'>No files uploaded yet</span>", unsafe_allow_html=True)
     
else:

    with stylable_container(
        key="scrollable_container",
        css_styles="""
            {
                max-height: 200px;
                overflow-y: scroll;
            }
            """,
    ):
        selected_file_names = st.sidebar.multiselect("select files",
            options=available_files,
            key="selected_file_radio",
            label_visibility="collapsed"
            )
        
    if len(selected_file_names) == 1:
        st.session_state.selected_file_name = selected_file_names[0]
    else:
        st.session_state.selected_file_name = None

    if st.session_state.selected_file_name is not None:
        selected_file_name=st.session_state.selected_file_name
        selected_file_info=st.session_state.processed_file_info.get(selected_file_name)
        chat_key=selected_file_name

        if selected_file_info is not None and selected_file_info["ingested"]==True:
            session_viewer_path=selected_file_info["pdf_viewer_path"]
    
            session_file_name=st.session_state.selected_file_name

            if chat_key not in st.session_state.chat_history:
                    st.session_state.chat_history[chat_key] = []
            session_chat_history = st.session_state.chat_history[chat_key]

            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.subheader(f"Viewing: {st.session_state.selected_file_name}")
                use_container_width=True
                
                pdf_viewer(
                        session_viewer_path,
                        width="100%",
                        height=700,
                        zoom_level="auto",
                        viewer_align="right",
                        show_page_separator=True,
                        )

            with col2:
                st.subheader("Chat with your file")
                with stylable_container(
                    key="scroll_chat_container",
                    css_styles="""
                    {
                         max-height: 70vh;
                         overflow-y: auto;
                         padding-right: 10px;
            
                    }
           """
                ):

                    for message in session_chat_history:
                         avatar="https://img.icons8.com/?size=100&id=0LnHUOCnYTrK&format=png&color=F25081" if message["role"] == "user" else "https://img.icons8.com/?size=100&id=100414&format=png&color=7950F2"
                         with st.chat_message(message["role"], avatar=avatar):
                              st.write(message["content"])
                    
                #testing new feature
                if st.session_state.waiting_for_approval:
                    
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    snapshot = rag_graph_app.get_state(config)
                    
                    # Get values
                    current_rewritten_query = snapshot.values.get("query", "")
                    original_query_val = snapshot.values.get("original_query", "")

                    with st.chat_message("assistant", avatar="https://img.icons8.com/?size=100&id=100414&format=png&color=7950F2"):
                        st.write("I'm having trouble finding good results.")
                        st.write("I suggest rewriting the query, or we can try searching more documents with your original query.")
                        
                        with st.form(key="approval_form"):

                            st.markdown("""
                                <style>
                                div[data-testid="stForm"] button {
                                    background-color: #7c52f4 !important;
                                    border-color: #7c52f4 !important;
                                }

                               
                                div[data-testid="stForm"] button p {
                                    color: #ffffff !important;
                                }

                               
                                div[data-testid="stForm"] button:hover {
                                    background-color: #a53254 !important;
                                    border-color: #a53254 !important;
                                }

                                
                                div[data-testid="stForm"] button:hover p {
                                    color: #ffffff !important;
                                }

                             
                                div[data-testid="stForm"] button:focus, 
                                div[data-testid="stForm"] button:active {
                                    background-color: #7c52f4 !important;
                                    border-color: #7c52f4 !important;
                                    color: #ffffff !important;
                                }
                        }
                                </style>""", unsafe_allow_html=True)
                            
                            new_query_input = st.text_input("Suggested Rewrite:", value=current_rewritten_query)
                            
                            
                            col_approve, col_orig = st.columns([1, 1])
                            
                            with col_approve:
                            
                                approve_clicked = st.form_submit_button("Approve Rewrite")
                            
                            with col_orig:
                                
                                original_clicked = st.form_submit_button("Use Original Query")
                        
                        if approve_clicked:
                            # 1. Update Query to the Input Box value
                            # 2. Set decision to 'retry' (goes to retrieve)
                            rag_graph_app.update_state(
                                config, 
                                {"query": new_query_input, "user_decision": "retry"}
                            )
                            st.session_state.chat_history[chat_key].append({"role": "assistant", "content": f"*(Retrying with: {new_query_input})*"})
                            run_rag_graph(input_state=None, chat_key=chat_key)
                            st.rerun()

                        elif original_clicked:
                            # 1. Revert Query to Original
                            # 2. Set decision to 'expand' (jumps to expand_retrieval)
                            rag_graph_app.update_state(
                                config, 
                                {"query": original_query_val, "user_decision": "expand"}
                            )
                            st.session_state.chat_history[chat_key].append({"role": "assistant", "content": "*(Reverting to original query and expanding search)*"})
                            run_rag_graph(input_state=None, chat_key=chat_key)
                            st.rerun()

                # CONDITION 2: Normal state (Not waiting) -> Show Chat Input
                else:
                    if user_input := st.chat_input("Ask a question about the document:"):
                        
                        # 1. Append user message
                        st.session_state.chat_history[chat_key].append({"role": "user", "content": user_input})
                        history_context = get_formatted_history(chat_key)
                        with st.spinner("Thinking..."):
                            # 2. Create Initial State
                            initial_state = GraphState(
                                query=user_input,
                                original_query=user_input, 
                                selected_file_names=[st.session_state.selected_file_name] if st.session_state.selected_file_name else list(selected_file_names),
                                search_index_name=search_index,
                                documents=[],
                                answer="",
                                relevance_score=0,
                                retry_count=0,
                                search_kwargs={"k": 5}
                            )
                        
                        # 3. Start New Run
                        run_rag_graph(input_state=initial_state, chat_key=chat_key)
                        
                        st.rerun()



    if len(selected_file_names) > 1 and st.session_state.selected_file_name is None:
         chat_key = tuple(sorted(selected_file_names))
         
         if chat_key not in st.session_state.chat_history:
                st.session_state.chat_history[chat_key] = []
         session_chat_history = st.session_state.chat_history[chat_key]

         st.subheader(f"Chat with: {', '.join(selected_file_names)}")
         
         # Display Chat History
         with stylable_container(
                    key="scroll_chat_container",
                    css_styles="""
                    {
                         max-height: 70vh;
                         overflow-y: auto;
                         padding-right: 10px;
                    }
           """
         ):
                     for message in session_chat_history:
                         avatar="https://img.icons8.com/?size=100&id=0LnHUOCnYTrK&format=png&color=F25081" if message["role"] == "user" else "https://img.icons8.com/?size=100&id=100414&format=png&color=7950F2"
                         with st.chat_message(message["role"], avatar=avatar):
                             st.write(message["content"])

    
         if st.session_state.waiting_for_approval:
             
             config = {"configurable": {"thread_id": st.session_state.thread_id}}
             snapshot = rag_graph_app.get_state(config)
             
             # Get values from state
             current_rewritten_query = snapshot.values.get("query", "")
             original_query_val = snapshot.values.get("original_query", "")

             with st.chat_message("assistant", avatar="https://img.icons8.com/?size=100&id=100414&format=png&color=7950F2"):
                 st.write("I'm having trouble finding good results.")
                 st.write("I suggest rewriting the query, or we can try searching more documents with your original query.")
                 
                 with st.form(key="approval_form_multi"):
                     
                     st.markdown("""
                         <style>
                        div[data-testid="stForm"] button {
                                    background-color: #7c52f4 !important;
                                    border-color: #7c52f4 !important;
                                }

                               
                                div[data-testid="stForm"] button p {
                                    color: #ffffff !important;
                                }

                                
                                div[data-testid="stForm"] button:hover {
                                    background-color: #a53254 !important;
                                    border-color: #a53254 !important;
                                }

                                
                                div[data-testid="stForm"] button:hover p {
                                    color: #ffffff !important;
                                }

                                
                                div[data-testid="stForm"] button:focus, 
                                div[data-testid="stForm"] button:active {
                                    background-color: #7c52f4 !important;
                                    border-color: #7c52f4 !important;
                                    color: #ffffff !important;
                                }
                         </style>""", unsafe_allow_html=True)
                     
                     new_query_input = st.text_input("Suggested Rewrite:", value=current_rewritten_query)
                     
                     col_approve, col_orig = st.columns([1, 1])
                     
                     with col_approve:
                         approve_clicked = st.form_submit_button("Approve Rewrite")
                     
                     with col_orig:
                         original_clicked = st.form_submit_button("Use Original Query")
                 
                 if approve_clicked:
                     # Update state to 'retry' with new query
                     rag_graph_app.update_state(
                         config, 
                         {"query": new_query_input, "user_decision": "retry"}
                     )
                     st.session_state.chat_history[chat_key].append({"role": "assistant", "content": f"*(Retrying with: {new_query_input})*"})
                     run_rag_graph(input_state=None, chat_key=chat_key)
                     st.rerun()

                 elif original_clicked:
                     # Update state to 'expand' with original query
                     rag_graph_app.update_state(
                         config, 
                         {"query": original_query_val, "user_decision": "expand"}
                     )
                     st.session_state.chat_history[chat_key].append({"role": "assistant", "content": "*(Reverting to original query and expanding search)*"})
                     run_rag_graph(input_state=None, chat_key=chat_key)
                     st.rerun()

         # CONDITION 2: Normal Chat Input (Multi-file)
         else:
             if user_input := st.chat_input("Ask a question about the files:"):

                st.session_state.chat_history[chat_key].append({"role": "user", "content": user_input})
                history_context = get_formatted_history(chat_key)

                with st.spinner("Thinking..."):
                    # Create Initial State (Ensure original_query is set)        
                    initial_state = GraphState(
                                    query=user_input,
                                    original_query=user_input, # <--- Added this
                                    selected_file_names=list(selected_file_names), 
                                    search_index_name=search_index,
                                    documents=[],
                                    answer="",
                                    relevance_score=0,
                                    retry_count=0,
                                    search_kwargs={"k": 5}
                                )

                    # Run graph using the helper (handles interrupts)       
                    run_rag_graph(input_state=initial_state, chat_key=chat_key)
                            
                st.rerun()

placeholder = st.empty()

if not selected_file_names:
    with placeholder:
        st.markdown(
            """
            <div style='text-align: center; padding: 60px 20px; color: #888;'>
                <h1 style='color: #F25081; font-size: 48px; margin-bottom: 10px;'>DynaBOT</h1>
                <img src='https://img.icons8.com/?size=100&id=100414&format=png&color=F25081' width='150' style='margin-bottom: 30px; margin-left: -20px;' />
                <h2>No documents selected</h2>
                <p>Please upload or select a file from the sidebar to get started.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    placeholder.empty()
    