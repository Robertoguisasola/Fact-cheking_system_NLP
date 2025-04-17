import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------
# 1. Cargar y Preprocesar Artículos
# -------------------------

def load_and_preprocess_articles(corpus_path):
    """
    Carga los archivos .txt de la ruta dada y los preprocesa.
    """
    articles = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # Preprocesar el texto (quitar caracteres no deseados)
                text = text.replace('\n', ' ').strip()
                articles.append(text)
    return articles

# -------------------------
# 2. Generación de Embeddings con Sentence-BERT
# -------------------------

def generate_embeddings(articles, model, device):
    """
    Genera los embeddings para los artículos usando el modelo cargado (Sentence-BERT).
    """
    embeddings = []
    for article in tqdm(articles):
        # Generar embedding para cada artículo
        embedding = model.encode(article)
        embeddings.append(embedding)
    
    return np.vstack(embeddings)

# -------------------------
# 3. Crear el Índice FAISS
# -------------------------

def create_faiss_index(embeddings):
    """
    Crea el índice FAISS para realizar búsquedas semánticas en los embeddings.
    """
    dimension = embeddings.shape[1]  # Número de dimensiones de los embeddings
    index = faiss.IndexFlatL2(dimension)  # Utilizando L2 para distancia euclidiana
    index.add(embeddings)
    return index

# -------------------------
# 4. Guardar el Índice y los Embeddings
# -------------------------

def save_faiss_index_and_embeddings(index, embeddings):
    """
    Guarda el índice FAISS y los embeddings en los archivos correspondientes.
    """
    faiss.write_index(index, "faiss_index.index")
    np.save("embeddings.npy", embeddings)
    print("Índice FAISS y embeddings guardados.")

# -------------------------
# 5. Función Principal (main)
# -------------------------

def main():
    # 1. Verificar si hay GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Cargar el modelo de embeddings (Sentence-BERT)
    model_name = 'all-MiniLM-L6-v2'  # Usamos Sentence-BERT
    model = SentenceTransformer(model_name)
    model.to(device)  # Mover el modelo a GPU/CPU

    # Obtener la ruta del directorio actual donde se ejecuta el script
    current_directory = os.getcwd()

    # Crear la ruta completa hacia la carpeta de los artículos
    corpus_path = os.path.join(current_directory, 'Data', 'World_War_II', 'Wikipedia', 'English')
    articles = load_and_preprocess_articles(corpus_path)
    
    # 3. Generar embeddings para los artículos
    embeddings = generate_embeddings(articles, model, device)
    
    # 4. Crear el índice FAISS
    index = create_faiss_index(embeddings)
    
    # 5. Guardar el índice FAISS y los embeddings
    save_faiss_index_and_embeddings(index, embeddings)

# -------------------------
# 6. Ejecutar el main
# -------------------------

if __name__ == "__main__":
    main()
