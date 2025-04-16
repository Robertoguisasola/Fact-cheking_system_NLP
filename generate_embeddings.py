import os
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
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
# 2. Generación de Embeddings
# -------------------------

def generate_embeddings(articles, model, tokenizer, device):
    """
    Genera los embeddings para los artículos usando el modelo cargado.
    """
    embeddings = []
    for article in tqdm(articles):
        inputs = tokenizer(article, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Mover los inputs a GPU/CPU

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Tomamos el promedio de los tokens
            embeddings.append(embedding)
    return np.vstack(embeddings)

# -------------------------
# 3. Crear el Índice FAISS
# -------------------------

def create_faiss_index(embeddings):
    """
    Crea el índice FAISS para realizar búsquedas semánticas en los embeddings.
    """
    dimension = embeddings.shape[1]
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
    
    # 2. Cargar el modelo y el tokenizador de embeddings
    model_name = "distilbert-base-uncased"  # Modelo de embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    # 3. Ruta a la carpeta con tus archivos .txt
    corpus_path = "path_to_your_wikipedia_articles"  # Reemplaza con la ruta correcta
    articles = load_and_preprocess_articles(corpus_path)
    
    # 4. Generar embeddings para los artículos
    embeddings = generate_embeddings(articles, model, tokenizer, device)
    
    # 5. Crear el índice FAISS
    index = create_faiss_index(embeddings)
    
    # 6. Guardar el índice FAISS y los embeddings
    save_faiss_index_and_embeddings(index, embeddings)

# -------------------------
# 6. Ejecutar el main
# -------------------------

if __name__ == "__main__":
    main()
