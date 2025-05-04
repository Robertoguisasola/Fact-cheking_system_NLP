import os
import json
import pickle
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load configuration from JSON
with open("config.json", "r", encoding="utf-8") as cf:
    config = json.load(cf)

topics   = config["topics"]
languages = config["languages"]

# Reading of txt files and sabe all text and metadata
# Iterate through each topic
for topic in topics:
    topic_folder = os.path.join("Data", topic)
    if not os.path.isdir(topic_folder):
        continue  # Skip if the topic folder doesn't exist

    # Iterate through each source inside the topic 
    for source in os.listdir(topic_folder):
        source_folder = os.path.join(topic_folder, source)
        if not os.path.isdir(source_folder):
            continue  # Skip if not a valid directory

        # Iterate through supported languages
        for lang in languages:
            lang_folder = os.path.join(source_folder, lang)
            if not os.path.isdir(lang_folder):
                continue  # Skip if language-specific folder is missing

            # Process each text file in the language folder
            for fname in os.listdir(lang_folder):
                file_path = os.path.join(lang_folder, fname)

                # Load only plain text files
                if os.path.isfile(file_path) and fname.lower().endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    raw_texts.append(text)  # Add file content to corpus

                    # Store structured metadata for traceability and filtering
                    metadatas.append({"topic": topic, "source": source, "language": lang, "filename": fname})

# Split data in chunks
# With a 50-character overlap to preserve context between chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = [] 

# Iterate through each raw text and its associated metadata
for i, text in enumerate(raw_texts):
    chunks = splitter.split_text(text)  # Split text into overlapping chunks
    
    # Create a Document object for each chunk, preserving metadata
    for chunk in chunks:
        # Assign the original metadata to each chunk.
        docs.append(Document(page_content=chunk, metadata=metadatas[i]))

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS DB
vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

# Saving of indexes and documents
vectorstore.save_local("faiss_langchain_index")
with open("documents_langchain.pkl", "wb") as f:
    pickle.dump(docs, f)
