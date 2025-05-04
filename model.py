import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from langdetect import detect
from bert_score import score
from transformers import AutoTokenizer, AutoModelForCausalLM

# Loading of indexes, documents and model
index = faiss.read_index("faiss_index.index")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
TOKEN = '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'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=TOKEN
)

def verificar_afirmacion(claim: str) -> str:
  """
    Verifies the truthfulness of a factual claim using a RAG (Retrieval-Augmented Generation) pipeline 
    and a large language model (LLM). The function performs multilingual support, retrieves context from 
    a vector store, generates a step-by-step reasoning, highlights the key fragment, outputs a verdict, 
    and evaluates the answer using BERTScore.

    Parameters
    ----------
    claim : str
        The factual claim to be verified. The language is automatically detected and supported for 
        Spanish, English, French, and German.

    Returns
    -------
    str
        A formatted string that includes:
          - Step-by-step reasoning
          - Key supporting or refuting fragment (or "None")
          - Final verdict ("True", "False", or "Not enough information")
          - Confidence level (based on vector similarity)
          - BERTScore metrics (Precision, Recall, F1)
    """
    # Language detection
    lang = detect(claim)
    if lang == 'es':
        lang_inst = "Responde en español."
    elif lang == 'en':
        lang_inst = "Answer in English."
    elif lang == 'fr':
        lang_inst = "Répondez en français."
    elif lang == 'de':
        lang_inst = "Antworten Sie auf Deutsch."
    else:
        lang_inst = "Answer in the same language as the question."

    # Embedding + searching
    q_emb = embedder.encode([claim])
    D, I = index.search(q_emb, 5)
    retrieved = [documents[i] for i in I[0]]
    avg_dist = D[0].mean()

    # Confidence tag
    if avg_dist < 0.4:
        conf = "Alta"
    elif avg_dist < 0.7:
        conf = "Media"
    else:
        conf = "Baja"

    context = "\n\n---\n\n".join(retrieved)

    # Prompt construction
    prompt = f"""### Afirmación a verificar:
    {claim}

    ### Contexto recuperado:
    {context}

    ### Instrucciones:
    Eres un asistente de verificación de hechos especializado en Segunda Guerra Mundial.
    {lang_inst}
    1. Analiza paso a paso si la afirmación está **respaldada**, **contradicha** o **no cubierta** por el contexto.
    2. Señala el **fragmento más relevante** que apoye o refute la afirmación (o "None" si no hay información).
    3. Emite un **veredicto final**: "True", "False" o "Not enough information".
    4. Fundamenta tu veredicto únicamente en los documentos proporcionados. No inventes datos.

    ### Formato de respuesta:
    - Razonamiento paso a paso:
    - Fragmento clave (o "None"):
    - Veredicto final (True / False / Not enough information):
    """

    # LLM generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # BERTScore computing
    P, R, F1 = score([answer], ["\n".join(retrieved)], lang=lang,
                     model_type="microsoft/deberta-xlarge-mnli", verbose=False)

    # Formated output
    return (
        f"{answer}\n\n"
        f"📈 Confianza: {conf} ({avg_dist:.2f})\n"
        f"BERTScore – P: {P.mean().item():.4f}, "
        f"R: {R.mean().item():.4f}, "
        f"F1: {F1.mean().item():.4f}"
    )