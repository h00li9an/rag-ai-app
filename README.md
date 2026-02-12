# rag-ai-app
RAG based chat application with AI/LLM support
Milvus lite as vector database
OpenAI SDK compatible API expose

#install
docker build -t rag-app .
docker run -it -p 8001:8001 rag-app
docker compose build
docker compose up
