# WWII Fact-Checking RAG System
This project implements a fact-checking system based on Retrieval-Augmented Generation (RAG). It uses open-source LLMs (Llama 3 Instruct) and a vector database (FAISS) to check the truthfulness of statements related to World War II.

## Project Structure

├── model.py                   # Main script: search and answer generation
├── wiki_api.py                 # Downloads Wikipedia documents by topic and language
├── utils/
│   ├── embeddings.py           # Generates embeddings and FAISS index
│   ├── veracity_checker.py     # (optional) Fact-checking logic modules
├── Data/
│   └── <topic>/Wikipedia/<language>/   # Downloaded Wikipedia .txt documents
├── utils/faiss_index.index     # FAISS index
├── utils/documents.pkl         # Chunked document texts

## How to Run the Project
### 1. Environment Setup
The project uses Anaconda to manage the environment, but some libraries must be installed separately via pip.

First, create the environment:

```
conda create --name rag_factcheck python=3.10
conda activate rag_factcheck
```
```
conda install --file environment.txt
```
Then, install additional libraries via pip:
```
pip install -r requirements.txt
```
This ensures that all libraries are properly installed.

### 2. Download documents from Wikipedia
Use wiki_api.py to download articles related to a specific topic:

```
python wiki_api.py
```

It will ask for:

    - Language (e.g., en for English, es for Spanish)

    - Topic (e.g., World War II)

Documents will be saved under folder:

Data/<topic>/Wikipedia/<language>/

### 3. Generate embeddings and build FAISS index
Run:
```
python utils/embeddings.py
```

This script:

    - Loads all .txt files from Data/
    - Splits texts into chunks
    - Computes sentence embeddings
    - Builds a FAISS vector index
    - Saves both the index (faiss_index.index) and document chunks (documents.pkl) in utils/

### 4. Start the fact-checking system
Run:
```
python model.py
```
This will:

    - Load the Llama-3-Instruct model
    - Perform a semantic search in the FAISS database
    - Generate a structured fact-checking answer
    - Translate automatically if the user input is in Spanish
    - Display the final answer, confidence score, and execution time

## Technologies Used
    - FAISS: Fast similarity search of embeddings
    - Sentence-Transformers: Sentence embedding generation
    - HuggingFace Transformers: LLM
    - LangDetect: Language detection
    - Prompt Engineering

## Features
    - Detects the question language automatically 
    - Answers in the same language as the input
    - Confidence score calculated based on FAISS distance
    - Tracks execution time per query
    - Fully modular and extendable to other topics and languages

