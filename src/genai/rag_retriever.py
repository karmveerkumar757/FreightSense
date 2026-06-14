# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import math
import re

# Global Chroma client and collection cached
_chroma_client = None
_collection = None

def get_chroma_collection():
    """
    Initializes the local ChromaDB client and creates/gets the 'regulations' collection.
    """
    global _chroma_client, _collection
    if _collection is None:
        db_path = os.path.join("vectordb", "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        
        print(f"📦 Connecting to ChromaDB at: {db_path}")
        _chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Initialize native embedding function
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Get or create collection
        _collection = _chroma_client.get_or_create_collection(
            name="regulations",
            embedding_function=emb_fn
        )
    return _collection

def add_documents_to_rag(documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
    """
    Adds chunks of text to the regulations collection in RAG vector store.
    """
    collection = get_chroma_collection()
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Added {len(documents)} document chunks to ChromaDB.")

def hybrid_rerank(query: str, documents: List[str], top_k: int = 3) -> List[str]:
    """
    Implements a Keyword Re-ranker on top of Semantic Search results.
    Boosts the score of chunks that contain exact keyword matches from the query.
    """
    if not documents:
        return []
        
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_docs = []
    
    for doc in documents:
        doc_lower = doc.lower()
        score = 0.0
        # Simple TF-IDF style heuristic
        for word in query_words:
            if len(word) > 3: # Ignore stopwords
                count = doc_lower.count(word)
                if count > 0:
                    score += math.log(count + 1) * 2.0 # Boost score
                    
        scored_docs.append((score, doc))
        
    # Sort by hybrid score descending, fallback to original semantic rank
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K
    return [doc for score, doc in scored_docs[:top_k]]

def retrieve_context(query: str, n_results: int = 3) -> List[str]:
    """
    Retrieves the top-N most relevant regulation chunks using Hybrid Search 
    (Semantic Search + Keyword Re-ranking).
    """
    collection = get_chroma_collection()
    
    # Check if database is empty
    count = collection.count()
    if count == 0:
        print("⚠️ ChromaDB regulations collection is empty. Returning empty context.")
        return []
        
    # Fetch top 10 semantic results first to broaden the pool
    semantic_fetch_count = min(10, count)
    results = collection.query(
        query_texts=[query],
        n_results=semantic_fetch_count
    )
    
    # Flatten the result list
    retrieved_docs = []
    if results and "documents" in results and results["documents"]:
        retrieved_docs = results["documents"][0]
        
    # Apply Keyword Re-ranking (Hybrid Search)
    if retrieved_docs:
        print(f"🔍 Hybrid Search: Re-ranking top {len(retrieved_docs)} semantic results...")
        retrieved_docs = hybrid_rerank(query, retrieved_docs, top_k=n_results)
        
    return retrieved_docs

def retrieve_and_generate(constraints: dict, use_agentic: bool = True) -> dict:
    """
    Unified entry point. Either uses the new Agentic RAG ReAct loop or the old single-pass logic.
    """
    if use_agentic:
        from src.genai.agentic_rag import AgenticRAG
        agent = AgenticRAG()
        return agent.run(constraints)
    else:
        # Fallback to single pass logic (for backward compatibility)
        query = " ".join(constraints.get("cargo_type", []) + constraints.get("locations", []))
        if not query:
            query = "Indian transport compliance"
            
        chunks = retrieve_context(query)
        from src.genai.prompt_builder import build_compliance_prompt
        import google.generativeai as genai
        import json
        
        prompt = build_compliance_prompt(constraints, chunks)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except:
            return {"overall_risk": "MEDIUM", "advisory_text": text}

def ingest_regulations_folder():
    """
    Scans the data/regulations folder for .txt or .pdf files, chunks them,
    and indexes them in ChromaDB.
    """
    reg_dir = os.path.join("data", "regulations")
    if not os.path.exists(reg_dir):
        print(f"⚠️ Regulations directory {reg_dir} does not exist.")
        return
        
    files = os.listdir(reg_dir)
    if not files:
        print(f"⚠️ No files found in {reg_dir} to ingest.")
        return
        
    print(f"📂 Scanning regulations folder for indexing: {reg_dir}...")
    
    import fitz  # PyMuPDF fallback
    chunks = []
    metadatas = []
    ids = []
    
    chunk_idx = 0
    for filename in files:
        file_path = os.path.join(reg_dir, filename)
        if filename.endswith(".txt"):
            print(f"📝 Indexing text file: {filename}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Chunk text by ~300 words
                words = content.split()
                for i in range(0, len(words), 300):
                    chunk_text = " ".join(words[i:i+300])
                    if len(chunk_text.strip()) > 50:
                        chunks.append(chunk_text)
                        metadatas.append({"source": filename, "type": "txt"})
                        ids.append(f"doc_{filename}_{chunk_idx}")
                        chunk_idx += 1
                        
        elif filename.endswith(".pdf"):
            print(f"📄 Indexing PDF file: {filename}")
            try:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page_text = doc[page_num].get_text()
                    # Chunk page text if too large
                    words = page_text.split()
                    for i in range(0, len(words), 300):
                        chunk_text = " ".join(words[i:i+300])
                        if len(chunk_text.strip()) > 50:
                            chunks.append(chunk_text)
                            metadatas.append({"source": filename, "page": page_num + 1, "type": "pdf"})
                            ids.append(f"doc_{filename}_p{page_num+1}_{chunk_idx}")
                            chunk_idx += 1
                doc.close()
            except Exception as e:
                print(f"❌ Failed to index PDF {filename}: {e}")
                
    if chunks:
        add_documents_to_rag(chunks, metadatas, ids)
    else:
        print("ℹ️ No new content extracted from regulation files.")
