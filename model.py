import faiss
import pickle
import torch
import os
from sentence_transformers import SentenceTransformer
from langdetect import detect
from bert_score import score
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
TOKEN = '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'


# Cargar FAISS y documentos
index = faiss.read_index("faiss_index.index")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# LLM
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=TOKEN
)

# RAG loop
while True:
    query = input("💬 Statement sobre WWII: ")
    start_time = time.time()

    # --- Detectar idioma ---
    language = detect(query)

    if language == 'es':
        language_instruction = "Responde en español."
    elif language == 'en':
        language_instruction = "Answer in English."
    elif language == 'fr':
        language_instruction = "Répondez en français."
    elif language == 'de':
        language_instruction = "Antworten Sie auf Deutsch."
    else:
        language_instruction = "Answer in the same language as the question."

    # --- Embedding + búsqueda ---
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, 5)
    retrieved_docs = [documents[i] for i in I[0]]

    avg_distance = D[0].mean()

    if avg_distance < 0.4:
        confidence = "High"
    elif avg_distance < 0.7:
        confidence = "Medium"
    else:
        confidence = "Low"

    context = "\n".join(retrieved_docs)
    prompt = f"""### User Statement:
{query}

### Retrieved Context:
{context}

### Instructions:
You are a historical fact-checking assistant specialized in World War II. 
{language_instruction}
1. Carefully read the retrieved context.
2. Analyze if the user's statement is **supported**, **contradicted**, or **not addressed** based on the context. Think step-by-step.
3. If there is evidence, highlight the **most relevant excerpt** supporting or contradicting the statement.
4. Provide a **clear final verdict**: "True", "False", or "Not enough information".
5. Justify the verdict based only on the retrieved documents. Do not invent facts.

### Response Format:
- Step-by-step reasoning:
- Main supporting excerpt (or "None" if not enough information):
- Final Verdict (True / False / Not enough information):
"""

    # --- Resto igual ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # BERTScore
    references = [context]  # the retrieved context
    candidates = [answer]   # the model's output

    print("\n🧠 Answer:")
    print(answer)

    print(f"\n📈 Confidence: {confidence} ({avg_distance:.2f})")

    os.system("huggingface-cli logout")

    P, R, F1 = score(
    candidates,
    references,
    lang=language,
    model_type="microsoft/deberta-xlarge-mnli",  # Alternativa muy buena y pública
    verbose=False
    )
    print(f"BERTScore - Precision: {P.mean().item():.4f}, Recall: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"\n⏱️ Time elapsed: {elapsed_time:.2f} minutes")
