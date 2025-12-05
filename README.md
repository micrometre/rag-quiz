# RAG Example with Ollama and ChromaDB

A minimal Retrieval-Augmented Generation (RAG) implementation using Ollama for embeddings and language models, with ChromaDB as the vector database.

## Overview

This project demonstrates how to build a simple RAG system that:
- Indexes documents using embeddings
- Retrieves relevant context based on queries
- Generates answers using a local language model

## Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running locally
- Required Ollama models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull phi3:mini
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running with the required models.

## Usage

Run the example:
```bash
python rag_example.py
```

The script will:
1. Index sample documents about Python, machine learning, RAG, and Ollama
2. Process the query "What is RAG?"
3. Retrieve relevant context from the indexed documents
4. Generate an answer using the phi3:mini model

## How It Works

1. **Document Indexing**: Documents are converted to embeddings using the `nomic-embed-text` model and stored in ChromaDB
2. **Query Processing**: The user's query is converted to an embedding using the same model
3. **Retrieval**: ChromaDB finds the most similar documents based on vector similarity
4. **Generation**: The retrieved context and query are sent to `phi3:mini` to generate a contextual answer

## Customization

- **Change the LLM**: Replace `phi3:mini` with any Ollama model (e.g., `llama3.2`, `mistral`)
- **Add more documents**: Extend the `documents` list with your own content
- **Adjust retrieval**: Modify `n_results` parameter to retrieve more or fewer documents
- **Use different embeddings**: Try other embedding models available in Ollama

## Dependencies

- `ollama`: Python client for Ollama
- `chromadb`: Vector database for storing and querying embeddings

## License

MIT

## Contributing

Feel free to open issues or submit pull requests with improvements!
