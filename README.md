
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
### Langgraph Dashboard Showcasing Flow
<img width="587" height="694" alt="image" src="https://github.com/user-attachments/assets/e4031744-e0fb-4642-9c5c-bf78eaf33130" />

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

I made a very basic version of this project beforehand during the summer break because it seemed like a very simple idea. There are so many online sites that come up that promise to let you ask questions about uploaded files etc and they're all paid. I didn't understand why because they seemed easy enough. As I've iterated over this project I've learnt a lot and I've also learnt how many little things are involved in implementing something like this, and how having a running application is completely different from having a jupyter notebook implementation with a local vector store instead of an external one. 

This project demonstates every core learning of this course.

**1. Prompting**
Four prompt templates guide the LLM at different stages: answer generation (with chat history injection and source citation logic, along with custom detailed instructions to make it conversational), quality evaluation of generated answer (using a 1-10 rubric with explicit groundedness/relevance/completeness criteria) which is used later to run the retry logic, query rewriting (for failed answers in an attempt to produce better ones by analyzing context failure patterns), along with query rewriting for web/wikipedia/arxiv search instead of a vector similarity search. Temperature tuning has been varied depending on the task

**2. Structured Output**
GraphState defines 16 strongly-typed fields including query/original_query (preserving intent for retry logic), external_context with a state reducer (operator.add) for aggregating parallel search results, relevance_score (1-10 quality metric), and research_mode (toggle for external tools). Metadata is sanitized to convert non-JSON types (ObjectIds, dates) to strings, ensuring serialization safety across MongoDB operations.

**3. Semantic Search**
Vector embeddings use HuggingFace's all-MiniLM-L6-v2 (384-dimensional), stored in MongoDB Atlas with top-k cosine similarity ranking. Retrieval is adaptive: it starts with k=5, as in top 5 similar documents are retrieved from mongodb, this is then increased upon retry when answer evaluation is poor. This makes an assumption that there might be relevant information to the query in other documents not retrieved. Pre-filtering by file_name ( it is defined in the vector search index as an additional field in mongodb) ensures responses are scoped. I've played around with the Chunking (size=1000, overlap=200) and tested it a couple times before deciding on these valuest that preserve context but are granular enough for good semantic search.

**4. Retrieval Augmented Generation (RAG)**
A four-stage pipeline: (1) retrieve documents via vector similarity search, (2) generate answers using retrieved documents while also keeping in mind sources in the metadata + external research when research mode is toggled on which conducts parallel search across different sources, (3) evaluate quality (pass if score ≥5), and (4) retry adaptively. Retries escalate: first attempt rewrites the query, second increases search scope, third falls back to failure. Human-in-the-loop  has been integrated at the rewrite stage which prompts the users to approve/edit suggestions or reject them to expand the original search instead.

**5. Tool Calling & Parallel Execution**
Three external tools run simultaneously when the research mode is toggled on: Tavily Web Search (good for current events, broad topics,etc. gives back top 3 results), Wikipedia (good for definitions, historical facts,etc. returns 1000-char summaries), and ArXiv (good for academic papers and research). This covers a large base of external sources to add as secondary sources to the user's document search. A query optimization node analyzes retrieved files and generates an exclusive prompt for the search queries thats different from the user prompt. It takes into consideration the context and writes a query that will yield the best search results from these external sources. Results aggregate into external_context via the state reducer before generation.

**6. LangGraph: State, Nodes, Graph**
A 12-node state machine is what compromises the entire workflow: retrieve → conditional research mode check → [optimize_query → parallel searches] or [direct generation] → evaluate → conditional quality routing → [pass, retry, or fail]. Conditional edges enable dynamic routing: research_mode toggles external search, relevance_score triggers retry logic, retry_count selects escalation strategy, user_decision (after human approval) chooses between query rewrite or search expansion. MemorySaver checkpointer enables thread-based persistence.

**7. Memory & Human-in-the-Loop**
Each session maintains a unique thread_id and chat histories invidually per file and per iteration of every selected combination of files. the last 2-4 messages (AI and Human Message pair form 1 message) are injected into generation and rewriting prompts ( number of messages injected is chosen dynamically based on the lenght of the coversation), enabling follow-up questions from the user. The graph pauses at human_approval before the retry node comes into action (when backend evaluation of generated answer before its actually displayed on the screen scores low so it reccomends the user to rewrite. It also displays the option of selecting an already re-written prompt to use), displaying the rewritten query in an editable form with two options: approve/edit (resulting in using the new query and then retrying retrieval with it) or reject (uses original query, expands search to k+5). State snapshots are checkpointed, allowing resumption after user decisions.

**LangSmith Integration**
Full tracing captures every LLM invocation (generation, evaluation, rewriting), vector search operations (embedding generation, similarity scores, retrieved chunks), tool calls (API responses), and state transitions.


## Plan

I plan to execute these steps to complete my project.

* [TODO] Step 1 involves implementing LangSmith Tracing across the project so that I can access tracing logs for the entire application in order to simplify debugging and optimization, and analyze usage in the future. 
  - [DONE]: Implemented langsmith tracing, did a test run and saw outputs in langsmith UI dashboard 

* [TODO] Step 2 involves FIXING a current error in the langgraph flow whenever the Retry nodes are run, it is failing to produce an expected output and falls back to the "failed" node
  - [DONE]: Fixed the errors
* [TODO] Step 3 involves modifying the current UI file structure and file handing in the background in order to group selected files into visual "folders" for querying. 
* [TODO] Step 4 involves adding source information of retrieved information in the answer
  - [DONE]: All mongodb documents and external documents have information retained such as file name/page number/url. There is a thorough process in place to retain and maintain all metadata. Implementation in UI however is not perfect yet and is being done in a very straightforward manner through the prompt. Have tried other methods but they were changing the way some of the documents were retrieved and hence messing with the RAG flow so I did not proceed further with that try. 
* [TODO] Step 5 involves adding a Research Mode, which on selecting, uses multiple data sources. Main priority still stands on uploaded sources, however, it conducts external web searches and depending on context, more specialized searches, for example, from google scholar, or ArXiv.
  - [DONE]: Have implimented external secondary search through a toggle button ("Research Node") in the UI that performs parallel tool calls for a general web search, wikipedia, and arXiv ( academic, mostly scientific articles). This covers a large number of bases for a variety of topics even if all tool calls are not utilized at the same time.
* [TODO] Step 6 is part of a larger step to involve tools and MCP. I want to add a tool that extracts and presents all tables from the uploaded document to the user upon asking. 
  - [DONE WITH CAVEAT]: Have technically implemented this step in a previous update that adds a thorough amount of refinement to data_processing in order to refine the process of extracting and reading tables. Did not implement a tool call for this in this iteration of the project. 
* [TODO] Step 7 involves integrating Memory and context management into the agent. The current system preserves history but has no memory. Add memory of past conversations.
  - [DONE]: Integrated short-term conversational memory that injects recent message history into the prompts for both generating answers and rewriting answers, enabling the model to handle follow-up questions. 
* [TODO] Step 8 involves improving langgraph structure.
  - [DONE]: Implemented a much more complicated langgraph structure that has a very thorough procedure for its Retry feature. When the AI fails to find a relevant answer and attempts to rewrite the query, the system now pauses execution to ask for human input instead of guessing automatically. There are two choices in the UI:
    - Approve/Edit: Accept (or tweak) the AI's suggested rewritten query to retry the search. If this fails still go to expand retrieval but with the rewritten query. 
    - Use Original & Expand: Reject the rewrite and force the system to use your original query, but search through more documents (increase k) to find the answer.
    - Made relevant UI changes to implement feature
* [TODO] Step 9 involves dynamic schema customization for the previously implemented folder functionality. When users upload documents, the LLM will automatically analyzes the content and infer document type and extract relevant metadata fields specific to that type (e.g., fiscal_year for financial docs, authors for research papers). This metadata is added to document chunks in MongoDB. This metadata is used for post-retrieval filtering, hence essentially making you able to do a refined Database search by simply uploading relevant files to a "folder" in the UI. (example: can be used to make something like snuGPT by simply uploading all prospectus documents and manuals)
* [TODO] Step 10 involves using Tesseract to expand from just pdf/pptx files to printed text OCR (scanned images)
* [TODO] Step 11 involves attempting to make a parser for NetCDF files to expand the project to accomodate scientific usecases
* [TODO] Step 12 involves making some implementations and use cases with the project to showcase
  - [DONE]: Have finalized on what documents to use to fully portray the potential of this project in the video
* [TODO] Step 13 UI changes for better aesthetic 
  - [DONE]: Changed some UI configurations with css for cleaner look
* [TODO] Step 14 modify prompt templates to give better outputs
  - [DONE]: Refined the generation and evaluation prompts to enforce stricter groundedness in retrieved documents, unbiased scoring criteria, and conditional topic suggestions.
* [TODO] Step 15 involves cleaning and restructuring code to fit in all these additional changes in a neater format without having bloated files with extremely lengthy codes
  - [DONE]: significantly restructured my entire codebase to include multiple new folders and more broken down and cleaner code
* [TODO] Step 16 revamp data pre processing
  - [DONE]:  Drastically reduced chunk_size and overlap in chunks to improve vector search granularity, yields better outputs.Modified logic to inject Markdown-formatted tables inline with their corresponding page text, preserving semantic proximity (previously appended to the end of the document list).Added heuristic checks (regex for alphanumeric content, empty DataFrame drops) to discard false-positive tables. Made changes to ensures text is captured even from PDFs where PyMuPDF fails which prevents "empty table" data from confusing the LLM and ensures table data is associated with the correct textual context.

## Video Summary Link: 

## Conclusion:

