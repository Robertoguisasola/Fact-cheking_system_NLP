import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 1. Leer archivos txt
texts = []
data_folder = 'Data/World_War_II/Wikipedia/English'

for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())

# 2. Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = []
for i, text in enumerate(texts):
    chunks = splitter.split_text(text)
    docs.extend([Document(page_content=chunk, metadata={"source": f"doc_{i}"}) for chunk in chunks])

# 3. Crear embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Crear base de datos FAISS
vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

# 5. Guardar índice y documentos
vectorstore.save_local("faiss_langchain_index")
with open("documents_langchain.pkl", "wb") as f:
    pickle.dump(docs, f)
