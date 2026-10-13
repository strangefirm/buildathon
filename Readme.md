# 🌾 Tamil Nadu Farmer Schemes Assistant

A production-grade, highly accurate Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **FAISS**, **Hugging Face Embeddings**, and **OpenAI**. Designed specifically for farmers and agricultural officers, this application provides grounded, zero-hallucination answers regarding Tamil Nadu government agricultural schemes, subsidies, loans, and eligibility criteria.

---

## 🚀 Advanced Techniques & Optimizations Used

To elevate this project from a standard prototype into an enterprise-ready, robust system, several advanced AI, retrieval, and LLMOps techniques were implemented:

1. **Hybrid Retrieval (Dense + Sparse Fusion)**:
   * Combines semantic vector similarity search (**FAISS**) with keyword matching (**BM25Okapi**).
   * Merges results using **Reciprocal Rank Fusion (RRF)** to ensure accurate lookups for broad semantic meanings as well as specific scheme names, acronyms, and codes (e.g., *PMFBY*).

2. **Cross-Encoder Reranking**:
   * Utilizes a fine-grained cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to deeply analyze query-document interactions.
   * Acts as a precision quality gatekeeper, filtering out noisy passages and passing only the top-ranked context chunks to the LLM.

3. **Multilingual Embeddings**:
   * Employs `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to map cross-lingual concepts into a shared vector space, facilitating seamless queries in both English and Tamil transliteration.

4. **Conversational Context & Query Resolution**:
   * Smart prompt engineering and history analysis resolve referential follow-up questions ("what about this scheme?"), maintaining multi-turn context continuity without losing grounding.

5. **Strict Grounding & Security Guardrails**:
   * **Input Guardrails**: Regex pattern matching intercepts and blocks prompt injections, system prompt leaks, and off-topic queries.
   * **Output Guardrails & Abstention Policy**: Enforces strict context-dependency, mandatory square-bracket source citations (`[1], [2]`), and a verified bilingual fallback message if documents lack the required answer.

6. **Asynchronous Evaluations (DeepEval)**:
   * Automatically measures **Faithfulness** (zero hallucination against context) and **Answer Relevancy** (directness to query).
   * Runs evaluations asynchronously in a background daemon thread to log performance metrics to `eval_log.jsonl` with **zero latency impact** on user chat streaming.

7. **Production Observability & Caching**:
   * **LangSmith Tracing**: Full end-to-end telemetry wrapper for OpenAI calls tracking tokens, latency, and retrieval accuracy.
   * **Index Fingerprinting**: SHA-256 document hashing caches the FAISS vector index locally (`index_store/`), avoiding expensive redundant re-embeddings on application restarts.

---

## 🌟 Key Architectural Features

* **Hybrid Retrieval Architecture**: Dense vector + sparse keyword matching via RRF.
* **Advanced Cross-Encoder Reranking**: Maximizes context precision before generation.
* **Bilingual Support**: Native handling for English and Tamil queries.
* **Asynchronous DeepEval & LangSmith**: Continuous quality tracking and telemetry.

---

## 🛠️ Tech Stack & Components

* **Frontend**: [Streamlit](https://streamlit.io/) with a customized agricultural theme.
* **Embeddings**: Hugging Face `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Vector Store**: FAISS (Facebook AI Similarity Search) with local index caching.
* **LLM**: OpenAI `gpt-4o-mini` with real-time token streaming (`st.write_stream`).
* **Evaluation & Tracing**: DeepEval & LangSmith.

---

## 📂 Project Directory Structure

```text
tn-farmer-schemes-assistant/
│
├── app.py                  # Main application script (Streamlit UI + RAG Engine)
├── agri25_26.pdf           # Primary official scheme document (Policy note / Budget)
├── data/
│   └── schemes/            # Additional optional scheme PDFs, .txt, or .md files
├── index_store/            # Cached FAISS index and chunk metadata (auto-generated)
├── assets/
│   └── tn_logo.png         # Government logo / branding icon
├── logs/
│   └── eval_log.jsonl      # Background DeepEval evaluation logs
└── .streamlit/
    └── secrets.toml        # API Keys configuration (OpenAI & LangSmith)
