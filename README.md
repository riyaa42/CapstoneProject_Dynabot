
# DynaBOT

=======

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Project report Template

## Title: Dynabot

## Overview: CURRENT STATUS 

This is a previosu project I have done and here is the current status 


**DynaBOT** is a Streamlit app for querying the content of uploaded PDF and PPTX files using a Retrieval-Augmented Generation (RAG) pipeline. 
It supports multi-file upload, intelligent chunking, MongoDB Atlas vector search, and a dynamic LLM query flow orchestrated using LangGraph.


## Features

### Dynamic File Handling

- **PDF and PPTX support**
- Extracts text and tables from PDFs (`pdfplumber`)
- Converts `.pptx` files to `.pdf` for inline viewing
- Single-file mode: Side-by-side viewer + chat
- Multi-file mode: Query across multiple files of different types
- Document metadata (like file name) stored for filtering during retrieval
- Supports live add/delete of documents from the vector store to sync with UI session

---

### Modular RAG Pipeline

- Uses LangGraph to define the flow:
  - **Retrieval → Generation → Evaluation → Retry → Fallback**
- Embeds user queries using HuggingFace models
- Performs top-k vector search using MongoDB Atlas
- Filters retrieved chunks by file name for scoped responses
- Evaluates the LLM answer quality on a 1–10 scale
- If quality is low:
  - Retry 1: Rewrites the query for clarity
  - Retry 2: Expands retrieval (increases `k`)
  - Fail: Falls back to default output
- Retry counter ensures clean loop exit

---

### Vector Store with MongoDB

- Stores embedded chunks in MongoDB Atlas using `MongoDBAtlasVectorSearch`
- Embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`
- Real-time addition/removal of documents to keep storage in sync with session

---

###  Streamlit Chat Interface

- File upload, viewing, and selection in sidebar
- Inline chat window updates based on selected file(s)
- Maintains chat history and response traceability
- UI layout adapts to single or multi-file mode

---

##Setup Instructions 

IMPORTANT!!!!!!⚠️

  
##1. LibreOffice should be installed in the system as streamlit ui doesnt support pptx files for viewing and i had to convert it to pdf for viewing purposes
      (program still processes .pptx files normally for rag processes it just converts for ui)

 macOS
      
      '''
      brew install --cask libreoffice  
      '''

windows

      '''
       https://www.libreoffice.org/download/
       '''
       
      
##2. path to soffice is hardcoded in data_processing.py in function "def convert_pptx_to_pdf" as the default soffice path in MacOS. if different path or 
          if on :
               windows ( C:\\Program Files\\LibreOffice\\program\\soffice.exe), 
               linux (soffice) 
          please edit path to aforementioned before running.
          will update for ease of access in future.
     

##1. Clone the repository
```
git clone https://github.com/yourusername/dynabot.git
cd dynabot
```

##2. Create a Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate
```
##3. Install Dependencies 
```
pip install -r requirements.txt
```

##4. Set up environment variables
```
cp .env.example .env
```
Edit the .env file and enter your actual values

```
LANGCHAIN_API_KEY=your_langchain_key
GOOGLE_API_KEY=your_google_key
MONGO_URL=your_mongodb_uri
DB_NAME=your_db_name
COLLECTION_NAME=your_collection
SEARCH_INDEX=your_index_name
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
(make sure your vector search index is set up on mongodb using the .json file in the repo. number of embeddings depend on the embedding model and are set in my .json according to minilm-l6-v2)

##5.Run the app
```
streamlit run app.py
```



## Reason for picking up this project

Explain how this project is aligned with this course content.


## Plan

I plan to execute these steps to complete my project.

- [TODO] Step 1 involves implementing LangSmith Tracing across the project so that I can access tracing logs for the entire application in order to simplify debugging and optimization, and analyze usage in the future. 
- [TODO] Step 2 involves FIXING a current error in the langgraph flow whenever the Retry nodes are run, it is failing to produce an expected output and falls back to the "failed" node
- [TODO] Step 3 involves modifying the current UI file structure and file handing in the background in order to group selected files into visual "folders" for querying. 
- [TODO] Step 4 involves adding source information of retrieved information in the answer
- [TODO] Step 5 involves adding a Research Mode, which on selecting, uses multiple data sources. Main priority still stands on uploaded sources, however, it conducts external web searches and depending on context, more specialized searches, for example, from google scholar, or ArXiv.
- [TODO] Step 6 is part of a larger step to involve tools and MCP. I want to add a tool that extracts and presents all tables from the uploaded document to the user upon asking. 
- [TODO] Step 7 involves utility to search files in the local system and Google Drive MCP and not just the uploaded one (MCP)
- [TODO] Step 8 involves adding token streaming
- [TODO] Step 8 involves integrating Memory and context management into the agent. The current system preserves history but has no memory. Add memory of past conversations and if needed store in the external database (MongoDB)
- [TODO] Step 9 involves improving langgraph structure and implementing sub graphs. 
- [TODO] Step 10 involves adding token streaming
- [TODO] Step 11 involves configuring all relevant files to use a free LLM model instead of gemini-2.0-flash
- [TODO] Step 12 involves dynamic schema customization for the previously implemented folder functionality. When users upload documents, the LLM will automatically analyzes the content and infer document type and extract relevant metadata fields specific to that type (e.g., fiscal_year for financial docs, authors for research papers). This metadata is added to document chunks in MongoDB. This metadata is used for post-retrieval filtering, hence essentially making you able to do a refined Database search by simply uploading relevant files to a "folder" in the UI. (example: can be used to make something like snuGPT by simply uploading all prospectus documents and manuals)

## Video Summary Link: 

## Conclusion:

