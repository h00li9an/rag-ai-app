import os
import uvicorn
import io
import json
import time
import uuid
import logging
import concurrent.futures
import asyncio
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pypdf import PdfReader
from docx import Document as DocxDocument

import openai
from google import genai
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# --- Configuration & Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")
CONFIG_FILE = "rag_config.json"

# Global State
class AppState:
    def __init__(self):
        self.milvus_client = None
        self.active_collection = "rag_default"
        # Cache for local model to prevent reloading
        self.local_embedding_model = None 
        self.local_model_name = ""
        
        # Default Config
        self.config = {
            # Inference Config
            "openai_key": os.getenv("OPENAI_API_KEY", ""),
            "gemini_key": os.getenv("GEMINI_API_KEY", ""),
            "openai_model": os.getenv("DEFAULT_OPENAI_MODEL", "gpt-3.5-turbo"),
            "gemini_model": os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash"),
            "active_stack": "openai", 
            
            # Embedding Config (Decoupled from Inference)
            "embedding_provider": "local", # openai, gemini, local
            "embedding_model": "all-MiniLM-L6-v2", 
            
            "rag_enabled": False,
            "rag_strict_mode": False
        }
        self.server_active = False
        
        # Load from disk if exists
        self.load_config()

    def load_config(self):
        """Loads configuration and state from disk."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    saved_data = json.load(f)
                    # Update config dict (merge to keep new defaults if any)
                    self.config.update(saved_data.get('config', {}))
                    # Restore server state logic
                    self.server_active = saved_data.get('server_active', False)
                    # Restore active collection
                    saved_collection = saved_data.get('active_collection')
                    if saved_collection:
                        self.active_collection = saved_collection
                logger.info("Configuration loaded from disk.")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

    def save_config(self):
        """Saves configuration and state to disk."""
        data = {
            'config': self.config,
            'server_active': self.server_active,
            'active_collection': self.active_collection
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Configuration saved to disk.")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def init_milvus(self):
        """Initializes or re-initializes the Milvus client."""
        if self.milvus_client:
            try:
                self.milvus_client.close()
            except Exception:
                pass
        try:
            self.milvus_client = MilvusClient("milvus_rag.db")
            logger.info("Milvus Lite (re)initialized.")
        except Exception as e:
            logger.error(f"Failed to init Milvus: {e}")
            self.milvus_client = None

state = AppState()

def get_local_model(model_name: str):
    """Singleton-like pattern for loading local models."""
    if state.local_embedding_model is None or state.local_model_name != model_name:
        logger.info(f"Loading local embedding model: {model_name}...")
        state.local_embedding_model = SentenceTransformer(model_name)
        state.local_model_name = model_name
        logger.info("Local model loaded.")
    return state.local_embedding_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Milvus Lite
    state.init_milvus()
    if not state.milvus_client:
        print("\n\033[91mCRITICAL: Milvus Lite failed to start. Ensure you have installed 'pymilvus[milvus_lite]'.\033[0m\n")
    
    # Ensure default collection exists if DB is up
    if state.milvus_client and not state.milvus_client.has_collection("rag_default"):
         pass

    yield
    # Shutdown
    if state.milvus_client:
        state.milvus_client.close()

app = FastAPI(title="GenAI RAG App", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ConfigUpdate(BaseModel):
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None
    openai_model: Optional[str] = None
    gemini_model: Optional[str] = None
    active_stack: Optional[str] = None
    
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    
    rag_enabled: Optional[bool] = None
    rag_strict_mode: Optional[bool] = None
    active_collection: Optional[str] = None

# --- Helper Functions ---

def parse_file_content(filename: str, file_bytes: bytes) -> str:
    content = ""
    filename = filename.lower()
    
    try:
        if filename.endswith(".pdf"):
            pdf_stream = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_stream)
            for page in reader.pages:
                text = page.extract_text()
                if text: content += text + "\n"
        elif filename.endswith(".docx"):
            docx_stream = io.BytesIO(file_bytes)
            doc = DocxDocument(docx_stream)
            for para in doc.paragraphs:
                content += para.text + "\n"
        elif filename.endswith(".txt"):
            content = file_bytes.decode("utf-8")
        elif filename.endswith(".csv"):
            csv_stream = io.BytesIO(file_bytes)
            df = pd.read_csv(csv_stream)
            content = df.to_string()
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise ValueError(f"Failed to parse file: {e}")
    
    return content

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    chunks = []
    start = 0
    if not text:
        return []
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def get_embedding(text: str) -> List[float]:
    """Generates embedding based on the configured embedding provider."""
    provider = state.config["embedding_provider"]
    model_name = state.config["embedding_model"]
    
    try:
        if provider == "openai":
            if not state.config["openai_key"]: raise ValueError("OpenAI Key missing")
            client = openai.OpenAI(api_key=state.config["openai_key"])
            response = client.embeddings.create(input=text, model=model_name)
            return response.data[0].embedding
            
        elif provider == "gemini":
            if not state.config["gemini_key"]: raise ValueError("Gemini Key missing")
            client = genai.Client(api_key=state.config["gemini_key"])
            result = client.models.embed_content(
                model=model_name,
                contents=text,
            )
            return result.embeddings[0].values
            
        elif provider == "local":
            model = get_local_model(model_name)
            embedding = model.encode(text)
            return embedding.tolist()
            
        else:
            raise ValueError("Invalid Embedding Provider")
    except Exception as e:
        logger.error(f"Embedding error ({provider}): {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a batch of texts."""
    provider = state.config["embedding_provider"]
    model_name = state.config["embedding_model"]
    
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=state.config["openai_key"])
            clean_texts = [t.replace("\n", " ") for t in texts]
            response = client.embeddings.create(input=clean_texts, model=model_name)
            return [data.embedding for data in response.data]
            
        elif provider == "gemini":
            client = genai.Client(api_key=state.config["gemini_key"])
            result = client.models.embed_content(
                model=model_name,
                contents=texts,
            )
            return [e.values for e in result.embeddings]
            
        elif provider == "local":
            model = get_local_model(model_name)
            embeddings = model.encode(texts)
            return embeddings.tolist()
            
        else:
            raise ValueError("Invalid Embedding Provider")
    except Exception as e:
        logger.error(f"Batch Embedding error ({provider}): {e}")
        raise e

def ensure_collection(dim: int, collection_name: str):
    """Ensures a collection exists with the correct dimension."""
    if not state.milvus_client:
        state.init_milvus()
        
    try:
        if state.milvus_client and state.milvus_client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} exists.")
            pass
        elif state.milvus_client:
            logger.info(f"Creating collection {collection_name} with dim {dim}")
            state.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                metric_type="COSINE",
                auto_id=True
            )
    except Exception as e:
        logger.warning(f"Error ensuring collection (attempting reconnect): {e}")
        state.init_milvus()
        if state.milvus_client and not state.milvus_client.has_collection(collection_name):
             state.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                metric_type="COSINE",
                auto_id=True
            )

def perform_rag_search(query: str) -> str:
    """Embeds query and searches Milvus."""
    try:
        if state.milvus_client is None:
            state.init_milvus()
            if state.milvus_client is None: return ""

        # Use the configured embedding provider
        embedding = get_embedding(query)
        
        coll_name = state.active_collection
        
        # Retry logic for search
        try:
            if not state.milvus_client.has_collection(coll_name):
                logger.warning(f"RAG search attempted on non-existent collection: {coll_name}")
                return ""

            res = state.milvus_client.search(
                collection_name=coll_name,
                data=[embedding],
                limit=3,
                output_fields=["text", "source"]
            )
        except Exception as e:
            logger.warning(f"Milvus search failed ({e}), reconnecting...")
            state.init_milvus()
            if not state.milvus_client: return ""
            
            res = state.milvus_client.search(
                collection_name=coll_name,
                data=[embedding],
                limit=3,
                output_fields=["text", "source"]
            )
        
        context_parts = []
        for hits in res:
            for hit in hits:
                text = hit["entity"].get("text", "")
                source = hit["entity"].get("source", "unknown")
                context_parts.append(f"[Source: {source}]\n{text}")
        
        return "\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"RAG Search failed: {e}")
        return ""

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the Single Page Application."""
    return HTML_TEMPLATE

@app.post("/api/start")
async def start_server(request: Request):
    """Activates the inference engine."""
    state.server_active = True
    state.save_config() # Persist state
    
    base_url = str(request.base_url).rstrip("/")
    endpoint_url = f"{base_url}/v1/chat/completions"
    
    logger.info(f"Inference Engine Started. External Endpoint: {endpoint_url}")
    
    return {
        "status": "active",
        "endpoint": endpoint_url,
        "message": "Inference Engine Started Successfully"
    }

@app.post("/api/stop")
async def stop_server():
    """Deactivates the inference engine."""
    state.server_active = False
    state.save_config() # Persist state
    logger.info("Inference Engine Stopped.")
    return {"status": "inactive", "message": "Inference Engine Stopped"}

@app.get("/api/collections")
async def get_collections():
    """Returns a list of available Milvus collections."""
    if state.milvus_client is None:
        state.init_milvus()
        if state.milvus_client is None: return {"collections": []}
        
    try:
        colls = state.milvus_client.list_collections()
        return {"collections": colls, "active": state.active_collection}
    except Exception as e:
        logger.warning(f"Error listing collections, reconnecting: {e}")
        state.init_milvus()
        try:
            if state.milvus_client:
                colls = state.milvus_client.list_collections()
                return {"collections": colls, "active": state.active_collection}
        except:
            pass
        return {"collections": []}

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Updates runtime configuration."""
    update_data = config.model_dump(exclude_unset=True)
    if "active_collection" in update_data:
        state.active_collection = update_data["active_collection"]
        logger.info(f"Active collection switched to: {state.active_collection}")
    
    state.config.update(update_data)
    state.save_config() # Persist state
    return {"status": "ok", "config": state.config}

@app.get("/api/config")
async def get_config():
    config_response = state.config.copy()
    config_response["server_active"] = state.server_active
    config_response["active_collection"] = state.active_collection
    return config_response

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...), 
    collection_name: Optional[str] = Form(None)
):
    """Handles file upload with streaming progress updates."""
    
    if state.milvus_client is None:
        raise HTTPException(status_code=503, detail="Vector DB Offline")

    # API Key check
    provider = state.config["embedding_provider"]
    if provider == "openai" and not state.config["openai_key"]:
        raise HTTPException(status_code=400, detail="OpenAI Key required")
    if provider == "gemini" and not state.config["gemini_key"]:
        raise HTTPException(status_code=400, detail="Gemini Key required")

    # 1. Read & Chunk
    try:
        file_bytes = await file.read() # Read async to avoid blocking
        text_content = parse_file_content(file.filename, file_bytes)
        
        if not text_content.strip():
             raise HTTPException(status_code=400, detail="File is empty")
             
        chunks = chunk_text(text_content)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text chunks from file")
    except Exception as e:
        logger.error(f"File read error: {e}")
        raise HTTPException(status_code=400, detail=f"File error: {str(e)}")

    async def process_generator():
        try:
            yield json.dumps({"status": "starting", "total_chunks": len(chunks)}) + "\n"

            # 2. Embed Test & Setup Collection
            try:
                # Basic embedding test
                test_emb = get_embedding("test")
                dim = len(test_emb)
            except Exception as e:
                yield json.dumps({"status": "error", "message": f"Embedding Test Failed: {str(e)}"}) + "\n"
                return

            # Determine Name
            target_collection = collection_name.strip() if collection_name and collection_name.strip() else f"rag_{provider}_{dim}"
            
            ensure_collection(dim, target_collection)
            state.active_collection = target_collection
            state.save_config() # Save new active collection

            yield json.dumps({"status": "setup_done", "collection": target_collection, "dimension": dim}) + "\n"

            # 3. Batch Processing & Incremental Insertion
            batch_size = 50 
            batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            total_batches = len(batches)
            
            start_time = time.time()
            completed_batches = 0
            
            # Helper for thread-safe embedding execution
            def process_batch_sync(batch_texts, filename):
                embeddings = get_batch_embeddings(batch_texts)
                return [{"vector": e, "text": t, "source": filename} for t, e in zip(batch_texts, embeddings)]

            loop = asyncio.get_running_loop()
            
            # Buffer for incremental inserts
            insert_buffer = []
            INSERT_THRESHOLD = 500

            for batch in batches:
                # Run embedding in thread pool to prevent blocking async loop
                try:
                    batch_data = await loop.run_in_executor(None, process_batch_sync, batch, file.filename)
                    insert_buffer.extend(batch_data)
                    completed_batches += 1
                    
                    # Incremental Insert to keep connection alive and prevent "too_many_pings" or massive payloads
                    if len(insert_buffer) >= INSERT_THRESHOLD:
                        try:
                            state.milvus_client.insert(collection_name=target_collection, data=insert_buffer)
                            insert_buffer = [] # Clear buffer
                        except Exception as e:
                            logger.warning(f"Incremental insert failed ({e}), reconnecting...")
                            state.init_milvus()
                            if state.milvus_client:
                                state.milvus_client.insert(collection_name=target_collection, data=insert_buffer)
                                insert_buffer = []

                    elapsed = time.time() - start_time
                    avg_time_per_batch = elapsed / completed_batches
                    remaining_batches = total_batches - completed_batches
                    eta_seconds = remaining_batches * avg_time_per_batch
                    
                    yield json.dumps({
                        "status": "progress", 
                        "processed": completed_batches, 
                        "total": total_batches,
                        "eta": round(eta_seconds, 1)
                    }) + "\n"
                    # Small yield to let event loop breathe if needed
                    await asyncio.sleep(0)

                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    # Continue to try other batches

            # 4. Insert Remaining
            yield json.dumps({"status": "inserting"}) + "\n"
            if insert_buffer:
                try:
                    state.milvus_client.insert(collection_name=target_collection, data=insert_buffer)
                except Exception as e:
                    logger.warning(f"Final insert failed ({e}), reconnecting...")
                    state.init_milvus()
                    if state.milvus_client:
                        state.milvus_client.insert(collection_name=target_collection, data=insert_buffer)
            
            yield json.dumps({"status": "complete", "count": len(chunks), "collection": target_collection}) + "\n"

        except Exception as e:
            logger.error(f"Stream Error: {e}")
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(process_generator(), media_type="application/x-ndjson")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible inference endpoint."""
    
    if not state.server_active:
        raise HTTPException(status_code=503, detail="Inference Engine is offline.")

    # 1. RAG
    context = ""
    last_user_msg = next((m for m in reversed(request.messages) if m.role == 'user'), None)
    
    rag_enabled = state.config["rag_enabled"]
    rag_strict = state.config.get("rag_strict_mode", False)

    if rag_enabled and last_user_msg:
        context = perform_rag_search(last_user_msg.content)
    
    # 2. Strict Mode Check
    if rag_enabled and rag_strict and not context:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": "I could not find any relevant information in the uploaded documents to answer your question."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 3. Messages & System Prompt
    final_messages = []
    
    system_instruction = "You are a helpful assistant."
    if context:
        if rag_strict:
            system_instruction = f"You are a helpful assistant. You MUST answer the user's question strictly based ONLY on the following context. If the answer is not contained within the context, you must state that you do not have the information.\n\nContext:\n{context}"
        else:
            system_instruction = f"You are a helpful assistant. Use the following context to answer the user's question if relevant:\n\nContext:\n{context}"
            
    final_messages.append({"role": "system", "content": system_instruction})
    
    for msg in request.messages:
        final_messages.append({"role": msg.role, "content": msg.content})

    # 4. Inference
    response_text = ""
    stack = state.config["active_stack"]
    model = request.model 
    
    try:
        if stack == "openai":
            if not state.config["openai_key"]: raise HTTPException(status_code=400, detail="OpenAI API Key missing")
            client = openai.OpenAI(api_key=state.config["openai_key"])
            completion = client.chat.completions.create(
                model=model,
                messages=final_messages,
                temperature=request.temperature
            )
            response_text = completion.choices[0].message.content

        elif stack == "gemini":
            if not state.config["gemini_key"]: raise HTTPException(status_code=400, detail="Gemini API Key missing")
            client = genai.Client(api_key=state.config["gemini_key"])
            prompt = ""
            prompt += f"System: {system_instruction}\n\n"
            for m in request.messages:
                prompt += f"{m.role.capitalize()}: {m.content}\n"
            
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            response_text = response.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

# --- Frontend HTML ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI RAG Chat</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .chat-scroll { height: calc(100vh - 250px); overflow-y: auto; }
        .overlay { background: rgba(17, 24, 39, 0.9); backdrop-filter: blur(4px); }
        progress { width: 100%; height: 8px; border-radius: 4px; }
        progress::-webkit-progress-bar { background-color: #374151; border-radius: 4px; }
        progress::-webkit-progress-value { background-color: #3b82f6; border-radius: 4px; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 flex h-screen overflow-hidden">

    <!-- Sidebar -->
    <div class="w-1/3 max-w-sm bg-gray-800 p-6 flex flex-col gap-6 border-r border-gray-700 overflow-y-auto">
        
        <div class="flex justify-between items-center mb-4">
            <h1 class="text-xl font-bold text-blue-400">Settings</h1>
            <span id="serverStatusBadge" class="text-xs font-bold px-2 py-1 rounded bg-red-600 text-white">OFFLINE</span>
        </div>

        <!-- INFERENCE CONFIG -->
        <div>
            <h2 class="text-sm font-bold text-gray-300 mb-2">Inference Stack</h2>
            <div class="bg-gray-700 p-3 rounded-lg mb-3">
                <label class="block text-xs font-semibold mb-1">Provider</label>
                <select id="stackSelect" class="w-full p-2 bg-gray-600 rounded border border-gray-500 text-white text-sm" onchange="updateInferenceUI()">
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Gemini</option>
                </select>
            </div>
            
            <div id="inferenceSettings" class="space-y-3">
                 <div>
                    <label class="text-xs uppercase text-gray-400">API Key</label>
                    <input type="password" id="apiKeyInput" class="w-full p-2 bg-gray-700 rounded border border-gray-600 text-xs">
                </div>
                <div>
                    <label class="text-xs uppercase text-gray-400">Model Name</label>
                    <input type="text" id="modelNameInput" class="w-full p-2 bg-gray-700 rounded border border-gray-600 text-sm">
                </div>
            </div>
        </div>

        <!-- EMBEDDING CONFIG -->
        <div class="border-t border-gray-600 pt-4 mt-2">
            <h2 class="text-sm font-bold text-purple-400 mb-2">Embedding Stack</h2>
            <div class="bg-gray-700 p-3 rounded-lg mb-3">
                <label class="block text-xs font-semibold mb-1">Provider</label>
                <select id="embedProviderSelect" class="w-full p-2 bg-gray-600 rounded border border-gray-500 text-white text-sm" onchange="updateEmbeddingUI()">
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Gemini</option>
                    <option value="local">Local (HuggingFace)</option>
                </select>
            </div>
            
            <div>
                <label class="text-xs uppercase text-gray-400">Embedding Model</label>
                <input type="text" id="embedModelInput" class="w-full p-2 bg-gray-700 rounded border border-gray-600 text-sm">
            </div>
        </div>

        <!-- RAG Config -->
        <div class="bg-gray-700 p-4 rounded-lg mt-auto">
            <div class="flex items-center justify-between mb-2">
                <span class="font-semibold text-sm">Enable RAG</span>
                <input type="checkbox" id="ragToggle" class="w-4 h-4">
            </div>
            
             <div class="flex items-center justify-between mb-2 pl-2 border-l-2 border-gray-600">
                <span class="text-xs text-gray-300">Strict Mode</span>
                <input type="checkbox" id="ragStrictToggle" class="w-4 h-4">
            </div>

            <div class="text-xs text-gray-400 mb-2">Vector DB: Milvus Lite</div>
            
            <!-- Collection Management -->
            <div class="mb-3 space-y-2">
                <label class="text-xs uppercase text-gray-400">Active Collection</label>
                <div class="flex gap-1">
                    <select id="collectionSelect" class="w-full p-2 bg-gray-600 rounded border border-gray-500 text-white text-xs" onchange="switchCollection()">
                        <!-- Populated by JS -->
                    </select>
                    <button onclick="fetchCollections()" class="p-1 bg-gray-600 rounded hover:bg-gray-500" title="Refresh">🔄</button>
                </div>
            </div>

            <!-- Upload -->
            <div class="border-t border-gray-600 pt-2">
                <label class="text-xs uppercase text-gray-400">New Database Name (Optional)</label>
                <input type="text" id="newCollectionName" class="w-full p-2 bg-gray-600 rounded border border-gray-500 text-white text-xs mb-2" placeholder="e.g. finance_docs">
                
                <label class="block mb-2 text-xs uppercase text-gray-400">Select File</label>
                <input type="file" id="fileUploadInput" class="w-full text-xs text-gray-300 file:mr-2 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700">
                
                <button onclick="uploadFile()" class="w-full mt-2 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm font-bold transition">
                    Upload & Index
                </button>
            </div>
            
            <!-- Progress -->
            <div id="progressContainer" class="hidden mt-3">
                 <div class="flex justify-between text-xs text-gray-300 mb-1">
                    <span id="progressText">Processing...</span>
                    <span id="etaText"></span>
                </div>
                <progress id="uploadProgress" value="0" max="100"></progress>
            </div>
        </div>
        
        <button onclick="saveConfig()" class="w-full py-2 bg-gray-600 hover:bg-gray-500 rounded font-bold transition text-xs">Save Settings</button>

        <div class="mt-4 border-t border-gray-600 pt-4">
            <div class="flex gap-2">
                <button id="startServerBtn" onclick="startServer()" class="flex-1 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-bold text-sm shadow-lg transition-all">
                    Start Server
                </button>
                <button id="stopServerBtn" onclick="stopServer()" class="flex-1 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-bold text-sm shadow-lg transition-all" disabled>
                    Stop
                </button>
            </div>
            
             <div id="apiInfoBox" class="hidden mt-4 bg-gray-900 p-3 rounded border border-green-500/30">
                <label class="text-xs uppercase text-green-400 font-bold block mb-1">External Endpoint</label>
                <div class="flex gap-2">
                    <input type="text" id="apiEndpointUrl" readonly class="w-full bg-transparent text-xs font-mono text-gray-300 focus:outline-none" value="...">
                    <button onclick="copyUrl()" class="text-gray-400 hover:text-white" title="Copy">📋</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col relative">
        <div id="offlineOverlay" class="absolute inset-0 z-50 overlay flex flex-col items-center justify-center text-center p-8">
            <div class="text-6xl mb-4">🤖</div>
            <h2 class="text-3xl font-bold text-white mb-2">Inference Engine Offline</h2>
            <p class="text-gray-400">Configure settings and click "Start Server" to begin.</p>
        </div>

        <header class="bg-gray-800 p-4 border-b border-gray-700 flex justify-between items-center">
            <div class="text-sm">
                Inference: <span id="dispInference" class="font-bold text-blue-400">...</span> | 
                Embedding: <span id="dispEmbedding" class="font-bold text-purple-400">...</span> |
                RAG: <span id="dispCollection" class="font-bold text-green-400">...</span>
            </div>
            <button onclick="clearChat()" class="text-xs text-red-400 hover:text-red-300">Clear</button>
        </header>

        <div id="chatContainer" class="chat-scroll p-6 space-y-4">
             <div class="flex justify-start">
                <div class="bg-gray-700 p-3 rounded-lg rounded-tl-none max-w-2xl">
                    <p class="text-sm">Ready.</p>
                </div>
            </div>
        </div>

        <div class="p-4 bg-gray-800 border-t border-gray-700">
            <div class="flex gap-2">
                <input type="text" id="userInput" disabled class="flex-1 p-3 bg-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50" placeholder="Type..." onkeypress="handleEnter(event)">
                <button id="sendBtn" onclick="sendMessage()" disabled class="p-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-bold px-6 transition disabled:opacity-50">Send</button>
            </div>
        </div>
    </div>

    <script>
        let config = {};
        const defaults = {
            openai: { model: 'gpt-3.5-turbo', embed: 'text-embedding-3-small' },
            gemini: { model: 'gemini-1.5-flash', embed: 'text-embedding-004' },
            local:  { embed: 'all-MiniLM-L6-v2' }
        };

        // Improved buffer handling for streamed JSON
        class JsonStreamReader {
            constructor() {
                this.buffer = '';
            }
            process(chunk) {
                this.buffer += chunk;
                const lines = [];
                let newlineIdx;
                while ((newlineIdx = this.buffer.indexOf('\\n')) !== -1) {
                    lines.push(this.buffer.slice(0, newlineIdx));
                    this.buffer = this.buffer.slice(newlineIdx + 1);
                }
                return lines;
            }
        }

        async function init() {
            try {
                const res = await fetch('/api/config');
                config = await res.json();
                
                document.getElementById('stackSelect').value = config.active_stack;
                document.getElementById('embedProviderSelect').value = config.embedding_provider;
                document.getElementById('embedModelInput').value = config.embedding_model;
                document.getElementById('ragToggle').checked = config.rag_enabled;
                document.getElementById('ragStrictToggle').checked = config.rag_strict_mode;
                
                updateInferenceUI(false);
                updateEmbeddingUI(false); 
                await fetchCollections();
                
                // If server was already active in persisted config, restore UI state
                if (config.server_active) {
                    // Re-enable UI manually since init happens before button click
                    const loc = window.location;
                    const endpoint = `${loc.protocol}//${loc.host}/v1/chat/completions`;
                    setServerActive(true, endpoint);
                }
                updateDisplay();
            } catch (e) { console.error(e); }
        }

        async function fetchCollections() {
            try {
                const res = await fetch('/api/collections');
                const data = await res.json();
                const select = document.getElementById('collectionSelect');
                select.innerHTML = '';
                
                if (data.collections.length === 0) {
                     const opt = document.createElement('option');
                     opt.text = "No collections";
                     select.appendChild(opt);
                } else {
                    data.collections.forEach(c => {
                        const opt = document.createElement('option');
                        opt.value = c;
                        opt.text = c;
                        select.appendChild(opt);
                    });
                    
                    if (data.active && data.collections.includes(data.active)) {
                        select.value = data.active;
                    } else if (config.active_collection && data.collections.includes(config.active_collection)) {
                         select.value = config.active_collection;
                    }
                }
                
            } catch(e) { console.error("Err fetching collections", e); }
        }
        
        async function switchCollection() {
            const newVal = document.getElementById('collectionSelect').value;
            if(newVal && newVal !== "No collections") {
                try {
                    const res = await fetch('/api/config', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ active_collection: newVal })
                    });
                    config = (await res.json()).config;
                    updateDisplay();
                } catch(e) { console.error(e); }
            }
        }

        function updateInferenceUI(reset = true) {
            const stack = document.getElementById('stackSelect').value;
            const keyInput = document.getElementById('apiKeyInput');
            const modelInput = document.getElementById('modelNameInput');
            
            if (stack === 'openai') {
                keyInput.value = config.openai_key || '';
                keyInput.placeholder = "sk-...";
                if(!reset && config.openai_model) modelInput.value = config.openai_model;
                else modelInput.value = defaults.openai.model;
            } else {
                keyInput.value = config.gemini_key || '';
                keyInput.placeholder = "AIza...";
                if(!reset && config.gemini_model) modelInput.value = config.gemini_model;
                else modelInput.value = defaults.gemini.model;
            }
        }

        function updateEmbeddingUI(reset = true) {
            const provider = document.getElementById('embedProviderSelect').value;
            const modelInput = document.getElementById('embedModelInput');
            if (reset) {
                if (provider === 'openai') modelInput.value = defaults.openai.embed;
                else if (provider === 'gemini') modelInput.value = defaults.gemini.embed;
                else if (provider === 'local') modelInput.value = defaults.local.embed;
            }
        }

        async function saveConfig() {
            const stack = document.getElementById('stackSelect').value;
            const embedProvider = document.getElementById('embedProviderSelect').value;
            
            const payload = {
                active_stack: stack,
                embedding_provider: embedProvider,
                embedding_model: document.getElementById('embedModelInput').value,
                rag_enabled: document.getElementById('ragToggle').checked,
                rag_strict_mode: document.getElementById('ragStrictToggle').checked
            };
            
            if (stack === 'openai') {
                payload.openai_key = document.getElementById('apiKeyInput').value;
                payload.openai_model = document.getElementById('modelNameInput').value;
                if(config.gemini_key) payload.gemini_key = config.gemini_key;
                if(config.gemini_model) payload.gemini_model = config.gemini_model;
            } else {
                payload.gemini_key = document.getElementById('apiKeyInput').value;
                payload.gemini_model = document.getElementById('modelNameInput').value;
                if(config.openai_key) payload.openai_key = config.openai_key;
                if(config.openai_model) payload.openai_model = config.openai_model;
            }

            const collVal = document.getElementById('collectionSelect').value;
            if(collVal && collVal !== "No collections") payload.active_collection = collVal;

            const res = await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            
            config = (await res.json()).config;
            updateDisplay();
            // Optional: alert("Settings Saved!"); // Removed to be less annoying on auto-save
        }

        async function startServer() {
            await saveConfig();
            const btn = document.getElementById('startServerBtn');
            btn.innerText = "Starting...";
            try {
                const res = await fetch('/api/start', { method: 'POST' });
                const data = await res.json();
                if (res.ok) setServerActive(true, data.endpoint);
            } catch (e) { alert("Error: " + e.message); }
            btn.innerText = "Start Server";
        }

        async function stopServer() {
            try {
                const res = await fetch('/api/stop', { method: 'POST' });
                if (res.ok) {
                    setServerActive(false);
                    alert("Server Stopped");
                }
            } catch(e) { alert("Error: " + e); }
        }

        function setServerActive(active, endpoint = "") {
            const overlay = document.getElementById('offlineOverlay');
            const startBtn = document.getElementById('startServerBtn');
            const stopBtn = document.getElementById('stopServerBtn');
            const statusBadge = document.getElementById('serverStatusBadge');
            
            if (active) {
                overlay.classList.add('hidden');
                document.getElementById('userInput').disabled = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('apiInfoBox').classList.remove('hidden');
                if (endpoint) document.getElementById('apiEndpointUrl').value = endpoint;
                
                startBtn.disabled = true;
                startBtn.classList.add('opacity-50', 'cursor-not-allowed');
                stopBtn.disabled = false;
                stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                
                statusBadge.innerText = "ONLINE";
                statusBadge.className = "text-xs font-bold px-2 py-1 rounded bg-green-600 text-white";
            } else {
                overlay.classList.remove('hidden');
                document.getElementById('userInput').disabled = true;
                document.getElementById('sendBtn').disabled = true;
                document.getElementById('apiInfoBox').classList.add('hidden');
                
                startBtn.disabled = false;
                startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                stopBtn.disabled = true;
                stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
                
                statusBadge.innerText = "OFFLINE";
                statusBadge.className = "text-xs font-bold px-2 py-1 rounded bg-red-600 text-white";
            }
        }

        function updateDisplay() {
            document.getElementById('dispInference').innerText = `${config.active_stack.toUpperCase()} (${config.active_stack === 'openai' ? config.openai_model : config.gemini_model})`;
            document.getElementById('dispEmbedding').innerText = `${config.embedding_provider.toUpperCase()}`;
            document.getElementById('dispCollection').innerText = config.active_collection || "None";
        }

        async function uploadFile() {
            const fileInput = document.getElementById('fileUploadInput');
            const file = fileInput.files[0];
            if (!file) {
                alert("Please select a file first.");
                return;
            }
            const collectionName = document.getElementById('newCollectionName').value;
            
            const formData = new FormData();
            formData.append('file', file);
            if(collectionName) formData.append('collection_name', collectionName);

            const progContainer = document.getElementById('progressContainer');
            const progBar = document.getElementById('uploadProgress');
            const progText = document.getElementById('progressText');
            const etaText = document.getElementById('etaText');
            progContainer.classList.remove('hidden');
            progBar.value = 0;
            progText.innerText = "Starting...";
            etaText.innerText = "";

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json();
                    alert("Upload Failed: " + err.detail);
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                const streamReader = new JsonStreamReader();
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, {stream: true});
                    const lines = streamReader.process(chunk);
                    
                    for (const line of lines) {
                        try {
                            const data = JSON.parse(line);
                            
                            if (data.status === 'starting') {
                                progText.innerText = `Preparing ${data.total_chunks} chunks...`;
                            } else if (data.status === 'progress') {
                                const percent = (data.processed / data.total) * 100;
                                progBar.value = percent;
                                progText.innerText = `Embedding Batch ${data.processed}/${data.total}`;
                                etaText.innerText = `ETA: ${data.eta}s`;
                            } else if (data.status === 'inserting') {
                                progText.innerText = "Inserting into DB...";
                                etaText.innerText = "";
                                progBar.value = 100;
                            } else if (data.status === 'complete') {
                                progText.innerText = "Done!";
                                alert(`Upload Complete! Added ${data.count} items to '${data.collection}'.`);
                                progContainer.classList.add('hidden');
                                fileInput.value = "";
                                document.getElementById('newCollectionName').value = "";
                                await fetchCollections(); 
                            } else if (data.status === 'error') {
                                alert("Error: " + data.message);
                            }
                        } catch (parseErr) { console.error("Parse Error", parseErr); }
                    }
                }
            } catch(e) { alert("Network Error: " + e); }
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if (!text) return;
            
            appendMessage('user', text);
            input.value = '';
            
            const model = config.active_stack === 'openai' ? config.openai_model : config.gemini_model;
            
            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: model,
                        messages: [{role: "user", content: text}]
                    })
                });
                const data = await res.json();
                if (res.ok) appendMessage('assistant', data.choices[0].message.content);
                else appendMessage('assistant', "Error: " + data.detail);
            } catch (e) { appendMessage('assistant', "Network Error"); }
        }

        function appendMessage(role, text) {
             const container = document.getElementById('chatContainer');
             const div = document.createElement('div');
             div.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
             div.innerHTML = `<div class="${role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-100'} p-3 rounded-lg max-w-2xl whitespace-pre-wrap">${text}</div>`;
             container.appendChild(div);
             container.scrollTop = container.scrollHeight;
        }

        function handleEnter(e) { if (e.key === 'Enter') sendMessage(); }
        function clearChat() { document.getElementById('chatContainer').innerHTML = ''; }
        function copyUrl() { navigator.clipboard.writeText(document.getElementById('apiEndpointUrl').value); }

        init();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
