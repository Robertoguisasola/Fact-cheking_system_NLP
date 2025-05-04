# WWII Fact-Checking RAG System
This repository implements a **Retrieval-Augmented Generation (RAG)** system for factual verification of claims related to **World War II**. It uses a hybrid pipeline combining open-source **LLMs** (Meta Llama 3 Instruct), **semantic search with FAISS**, and **BERTScore-based evaluation** to generate transparent and multilingual verdicts.

This project was developed as part of the "Machine Learning for Health" Master's Program (UC3M) — *Natural Language Processing Final Project 2025*.

---

## ✨ Key Features

* Multilingual support: answers in **English, Spanish, French, or German**.
* Uses **wikipedia + Scopus** as sources.
* Produces **structured justifications** with fragment citation.
* Outputs **BERTScore metrics** and **confidence level** based on FAISS distance.
* Optional DSPy module for programmatic chaining and signature optimization.

---

## 🤹 Authors

> *(Replace the list below with the names of your team members)*

* Elena Almagro Azor
* Mario Golbano Corzo
* Roberto Martínez - Guisasola Guerrero
* Juan Muñóz Villalón

---

## 🎓 Project Overview

This fact-checking system addresses the problem of LLM hallucinations by grounding generations in trusted sources. The system:

1. Retrieves top relevant documents via semantic search (FAISS).
2. Constructs a prompt incorporating retrieved context and the claim.
3. Uses **Llama 3.2 1B Instruct** for step-by-step reasoning and verdict.
4. Evaluates factual alignment with retrieved context using **BERTScore**.

The design meets the **mandatory functional requirements**:

* Cites supporting evidence.
* Answers only when information is found.
* Provides clear verdicts: **True**, **False**, or **Not enough information**.
* Responds in the input language.

---

## 📚 Directory Structure

```
├── model.py                   # Main script: search and answer generation
├── wiki_api.py                 # Downloads Wikipedia documents by topic and language
├── utils/
│   ├── embeddings.py           # Generates embeddings and FAISS index
│   ├── veracity_checker.py     # (optional) Fact-checking logic modules
├── Data/
│   └── <topic>/Wikipedia/<language>/   # Downloaded Wikipedia .txt documents
├── utils/faiss_index.index     # FAISS index
├── utils/documents.pkl         # Chunked document texts


## ⚙️ Setup Instructions

### 1. Create Environment

```bash
conda create -n rag_factcheck python=3.10
conda activate rag_factcheck
pip install -r requirements.txt
```

### 2. Prepare `config.json`

Configure:

```json
{
  "topics": ["World War II"],
  "languages": ["en", "es"],
  "api_key_scopus": "<YOUR_KEY>",
  "user_agent_wiki": "<username/email>",
  "max_articles_scopus": 50
}
```

The values correspond to:
* api_key_scopus: your scopus API key.
* topics: topics to be used in the system. Put each of them in inverted commas and separate them with a comma.
* languages: languages in which you want to download the information. It is independent of the languages in which the application is used. For English, put ‘en’. For Spanish, put ‘es’.
* max_articles_scopus: number of scientific articles to download from Scopus.
* user_agent_wiki: username and email to connect to the Wikipedia API.

### 3. Download the Corpus

```bash
python data_download.py
```

This fetches Wikipedia + Scopus documents for configured topics and languages.

### 4. Generate Embeddings & Build Vector Store

```bash
python embeddings.py
```

* Uses SentenceTransformers (`all-MiniLM-L6-v2`)
* Splits text into overlapping 500-char chunks
* Saves FAISS index and document metadata

### 5. Run the System (CLI or GUI)

```bash
python model.py 
python gui.py 
```

---

## 🔮 Example Output (CLI)

```
Claim: "The Normandy landings occurred in 1945."

- Step-by-step reasoning: [...]
- Key fragment: "The Allied invasion of Normandy took place on June 6, 1944 [...]"
- Final verdict: False

📊 Confidence: High (0.31)
BERTScore – P: 0.8651, R: 0.8492, F1: 0.8571
```

---

## 📊 Evaluation & Metrics

The system incorporates:

* **Recall\@K** (implicit via top-5 semantic search)
* **BERTScore** (DeBERTa for multilingual factual alignment)
* **Confidence estimation** based on FAISS average distance

---

## 📄 Technologies

* **FAISS** (vector store)
* **SentenceTransformers** (embeddings)
* **Transformers** (Llama 3, tokenizer)
* **LangChain + HuggingFaceEmbeddings** (optional DSPy integration)
* **Tkinter** (GUI)

---

## 💡 Extensions / Extras

* GUI with Tkinter
* DSPy-compatible `VeracityChecker` module
* Automatic translation of claims
* Multilingual corpus ingestion (Wikipedia + Scopus)

---

## ⚠️ Limitations

* Domain restricted to WWII, but it's prepared to use more topics
* No fine-tuning (uses prompt-only control)
* FactScore not implemented due to token constraints
* LLM may still hallucinate if context is poor

---

## 🚜 Future Work

* Integrate FactScore and MRR for formal evaluation
* Add CoT prompting or decomposition techniques
* Support hybrid corpus (news + academic + Wikipedia)
* Improve GUI with confidence sliders and summary view

---

## 📅 Project Context

This system was developed for the NLP Final Project of the "Machine Learning for Health" Master's Program (UC3M), under the theme **Fact-Checking System**.

All documents, models, and prompts comply with the functional and technical requirements of the assignment.

---

## 🔗 Resources

* [Meta Llama Models](https://huggingface.co/meta-llama)
* [FAISS Documentation](https://github.com/facebookresearch/faiss)
* [BERTScore Paper](https://arxiv.org/abs/1904.09675)
* [Wikipedia API](https://pypi.org/project/Wikipedia-API/)