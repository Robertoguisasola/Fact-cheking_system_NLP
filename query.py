import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from generate_embeddings import load_and_preprocess_articles
import os

# ------------------------
# 1. Configurar dispositivo (GPU/CPU)
# ------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# 2. Cargar el modelo de embeddings (Sentence-BERT)
# ------------------------

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Cargar el modelo de Sentence-BERT
embedding_model.to(device)  # Mover el modelo a GPU/CPU

# --------------------------
# 3. Cargar el modelo generativo (T5) para la respuesta
# --------------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_t5 = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model_t5.to(device)  # Asegúrate de mover el modelo T5 a GPU/CPU

# ------------------------
# 4. Cargar el índice FAISS desde el archivo guardado
# ------------------------

def load_faiss_index(index_path):
    """
    Carga el índice FAISS desde el archivo guardado.
    """
    return faiss.read_index(index_path)

# ------------------------
# 5. Función para realizar la búsqueda con FAISS
# ------------------------

def search(query, index, k=3):
    """
    Función que busca los k documentos más relevantes para una consulta dada.
    """
    # Generar el embedding para la consulta usando Sentence-BERT
    query_embedding = embedding_model.encode(query)

    # Realizar la búsqueda en FAISS
    distances, indices = index.search(np.array([query_embedding]), k)  # Recuperar los k documentos más cercanos

    current_directory = os.getcwd()

    # Crear la ruta completa hacia la carpeta de los artículos
    corpus_path = os.path.join(current_directory, 'Data', 'World_War_II', 'Wikipedia', 'English')
    articles = load_and_preprocess_articles(corpus_path)

    # Obtener los documentos relevantes
    relevant_docs = [articles[i] for i in indices[0]]  # Recupera los artículos más relevantes
    return relevant_docs

# ------------------------
# 6. Generar respuesta con T5 basada en los documentos recuperados
# ------------------------

def generate_answer(query, relevant_docs):
    """
    Función que genera una respuesta utilizando el modelo T5 basado en los documentos recuperados.
    """
    # Concatenar los documentos relevantes con la pregunta para crear el prompt
    input_text = query + " " + " ".join(relevant_docs)

    # Tokenizar y generar la respuesta
    inputs = tokenizer_t5(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_t5.generate(inputs['input_ids'], max_length=150)

    # Decodificar la respuesta generada
    answer = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
    return answer

# ------------------------
# 7. Función de interacción
# ------------------------

def interactive_mode():
    print("Welcome to the Fast-Checking System!")
    print("You can ask questions about World War II, climate change, 5G, and more. Type 'exit' to quit.")

    while True:
        # Hacer la pregunta
        query = input("Enter your question: ")

        if query.lower() == 'exit':
            print("Exiting the system.")
            break

        index_path = "faiss_index.index"  # Asegúrate de que este archivo exista en tu carpeta
        index = load_faiss_index(index_path)

        # Buscar documentos relevantes en el índice FAISS
        relevant_docs = search(query, index)

        # Si no se recuperaron documentos relevantes
        if len(relevant_docs) == 0:
            print(f"System: I'm sorry, I do not have enough information in the corpus to verify this claim.\n")
            continue

        # Generar una respuesta basada en los documentos recuperados
        answer = generate_answer(query, relevant_docs)

        # Si la respuesta está "verificada" o "falsa" según el modelo
        if "true" in answer.lower():
            verification = "True"
        elif "false" in answer.lower():
            verification = "False"
        else:
            verification = "I couldn't determine the truth value from the data."

        print(f"System: {verification}.")
        print("Justification:")
        for doc in relevant_docs:
            print(f"- {doc}")

        print("\nWhat else would you like to check?\n")

# ------------------------
# 8. Ejecutar el main
# ------------------------

if __name__ == "__main__":
    interactive_mode()
