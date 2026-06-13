# Agentic RAG Document Assistant

A production-ready full-stack application featuring agentic Retrieval-Augmented Generation (RAG) capabilities. Upload Excel and PDF documents, and query them using AI-powered agents.

## Features

- **Multi-document Upload**: Support for PDF and Excel files
- **Vector Search**: MongoDB-powered vector search for efficient document retrieval
- **Agentic RAG**: AI agents that reason and generate responses based on document content using local Ollama models
- **Modern Frontend**: Next.js-based user interface
- **Scalable Backend**: Flask API with modular architecture

## Tech Stack

- **Backend**: Python Flask, MongoDB, LangChain, Ollama (deepseek-r1:14b, nomic-embed-text)
- **Frontend**: Next.js, React, Tailwind CSS
- **Database**: MongoDB with vector search
- **AI/ML**: Ollama embeddings and LLM

## Prerequisites

- Python 3.8+
- Node.js 18+
- MongoDB (local or Atlas)
- Ollama installed with models: `ollama pull deepseek-r1:14b` and `ollama pull nomic-embed-text`

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your MongoDB URI
3. Ensure Ollama is running locally
4. For development:
   - Backend: `cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python run.py`
   - Frontend: `cd frontend && npm install && npm run dev`
5. For low-disk development on Windows (recommended for your setup):
  - Run both services: `START_APP_LIGHT.bat`
  - Run backend only: `START_BACKEND_LIGHT.bat`
  - Run frontend only: `START_FRONTEND_LIGHT.bat`
  - Lightweight mode keeps temp/cache data on the project drive and avoids heavy install/cache operations on each run.
6. For production: `docker-compose up --build`

## API Endpoints

- `POST /api/upload`: Upload documents
- `POST /api/query`: Query documents using RAG

## MongoDB Document Structure

Documents are stored in the `documents` collection with the following JSON structure:

```json
{
  "_id": "ObjectId",
  "filename": "document.pdf",
  "content": "Extracted text content...",
  "embedding": [0.1, 0.2, ..., 0.768]
}
```

## Usage

1. Upload PDF or Excel files through the frontend
2. Ask questions about the uploaded documents
3. Receive AI-generated responses based on document content using local Ollama models