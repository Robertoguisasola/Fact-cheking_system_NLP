import os
import json
import pickle
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Cargar configuración
with open("config.json", "r", encoding="utf-8") as cf:
    config = json.load(cf)

topics   = config["topics"]
languages = config["languages"]

# 1. Leer archivos txt y almacenar textos + metadatos
raw_texts = []
metadatas = []
for topic in topics:
    topic_folder = os.path.join("Data", topic)
    if not os.path.isdir(topic_folder):
        continue
    for source in os.listdir(topic_folder):
        source_folder = os.path.join(topic_folder, source)
        if not os.path.isdir(source_folder):
            continue
        for lang in languages:
            lang_folder = os.path.join(source_folder, lang)
            if not os.path.isdir(lang_folder):
                continue
            for fname in os.listdir(lang_folder):
                file_path = os.path.join(lang_folder, fname)
                if os.path.isfile(file_path) and fname.lower().endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    raw_texts.append(text)
                    metadatas.append({
                        "topic": topic,
                        "source": source,
                        "language": lang,
                        "filename": fname
                    })


# 2. Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for i, text in enumerate(raw_texts):
    chunks = splitter.split_text(text)
    for chunk in chunks:
        docs.append(Document(
            page_content=chunk,
            metadata=metadatas[i]
        ))

# 3. Crear embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Crear base de datos FAISS
vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

# 5. Guardar índice y documentos
vectorstore.save_local("faiss_langchain_index")
with open("documents_langchain.pkl", "wb") as f:
    pickle.dump(docs, f)
