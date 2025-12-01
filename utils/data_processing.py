from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredPowerPointLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
import pdfplumber
import pandas as pd
import subprocess


def clean_text(text: str) -> str:
    """Removes excessive whitespace and newlines."""
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def load_file(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    print(f"[DEBUG] Loading file: {file_path}")

    try:
        if ext == ".pdf":
            # STRATEGY 1: Try PyMuPDF
            loader = PyMuPDFLoader(file_path)
            raw_docs = loader.load()
            
            # Check if PyMuPDF actually found text
            total_text_len = sum([len(d.page_content) for d in raw_docs])
            print(f"[DEBUG] PyMuPDF found {total_text_len} characters.")

            use_plumber_for_text = False
            if total_text_len < 10:
                print("[WARN] PyMuPDF failed to extract meaningful text. Switching to pdfplumber text extraction.")
                use_plumber_for_text = True

            final_docs = []

            try:
                with pdfplumber.open(file_path) as pdf:
                    
                    num_pages = len(pdf.pages)
                    
                    for i in range(num_pages):
                        plumber_page = pdf.pages[i]
                        
                    
                        if use_plumber_for_text:
                            # Use plumber to extract text (slower but handles weird encodings better)
                            page_text = plumber_page.extract_text() or ""
                        else:
                            # Use the text we already got from PyMuPDF
                            if i < len(raw_docs):
                                page_text = raw_docs[i].page_content or ""
                            else:
                                page_text = ""

                       
                        # Find tables
                        tables = plumber_page.find_tables({
                            "vertical_strategy": "text", 
                            "horizontal_strategy": "lines"
                        })
                        
                        table_texts = []
                        for table in tables:
                            try:
                                table_data = table.extract()
                                if not table_data or len(table_data) < 2: continue
                                
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
                                if df.empty: continue
                                
                                # Ghost Table Check
                                table_str = df.to_string()
                                if len(re.findall(r'[a-zA-Z0-9]', table_str)) < 10: continue

                                # Markdown
                                markdown = df.to_markdown(index=False)
                                table_texts.append(f"\n[TABLE DATA]:\n{markdown}")
                            except:
                                continue 
                        
                        # Combine Text + Tables
                        if table_texts:
                            page_text += "\n" + "\n".join(table_texts)

                        # Create Document
                        doc = Document(
                            page_content=clean_text(page_text),
                            metadata={"source": file_path, "page": i + 1}
                        )
                        final_docs.append(doc)

              
                final_len = sum([len(d.page_content) for d in final_docs])
                print(f"[DEBUG] Final extraction length: {final_len}")

                if final_len == 0:
                     raise ValueError("Extracted text is empty (File might be a scanned image).")

                return final_docs

            except Exception as e:
                print(f"[ERROR] Detailed processing failed: {e}")
            
                # If we were using PyMuPDF and plumber crashed on tables, return raw PyMuPDF
                if not use_plumber_for_text and raw_docs:
                     print("Returning raw PyMuPDF text as fallback.")
                     return raw_docs
                return []

    
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
            raw_docs = loader.load()
            cleaned_docs = []
            for doc in raw_docs:
                doc.page_content = clean_text(doc.page_content)
                if "page_number" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page_number"]
                doc.metadata["source"] = file_path
                cleaned_docs.append(doc)
            return cleaned_docs

        else:
            raise ValueError(f"[ERROR] Unsupported file extension: {ext}")
    
    except Exception as e:
        print(f"[CRITICAL] load_file failed: {e}")
       
        return []

def split_docs(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Reduced chunk size to 1000 for better RAG context retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(docs)


def convert_pptx_to_pdf(pptx_path: str, output_dir: str) -> str:
    """
    Converts PPTX to PDF for the Streamlit Viewer (UI purposes only).
    """
    if not os.path.exists(pptx_path):
        return None
    
    file_name = os.path.splitext(os.path.basename(pptx_path))[0]
    pdf_path = os.path.join(output_dir, f"{file_name}.pdf")
    
    if os.path.exists(pdf_path):
        return pdf_path
    
    try:
       
        soffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
        if not os.path.exists(soffice_path):
             soffice_path = "soffice" # Global command fallback

        cmd = [
            soffice_path,
            "--headless",
            "--invisible",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            pptx_path,
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if os.path.exists(pdf_path):
            return pdf_path
        else:
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error converting {pptx_path} to PDF: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        return None
#loader=PyMuPDFLoader(file_path)



