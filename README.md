# ⚖️ LawAI India

> **AI-Powered Indian Legal Research Assistant**  
> A production-grade Retrieval-Augmented Generation (RAG) system for Indian law.

---

## 📋 Project Overview

LawAI India is a fully local, privacy-first legal AI assistant that can answer questions about Indian law across all major acts and codes. It uses a true RAG pipeline — never keyword search, never prompt-stuffing — to retrieve semantically relevant legal provisions and generate accurate, cited answers.

Supports questions in **English**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)**.

---

## ✨ Features

| Feature | Details |
|---|---|
| True RAG Pipeline | Embed → Vector Search → Retrieve → Generate |
| 20+ Indian Laws | Constitution, BNS, BNSS, BSA, IT Act, GST, and more |
| Multilingual | English / Hindi / Telugu |
| Local & Private | Runs entirely on your machine via Ollama |
| Streaming Output | Token-by-token response display |
| Legal Citations | Act name, section, page number per answer |
| Upload & Index | Drop any PDF; auto-indexed within seconds |
| Conversation Memory | Multi-turn context across the session |
| Export to PDF | Download AI responses as formatted PDFs |
| Download Chat | Export full chat history as JSON |
| Auto-Indexing | Monitor `/data/new_laws/` for new PDFs |
| Dark Legal Theme | Professional dark UI with gold accents |

---

## 🏗️ Architecture

```
User Question (English / Hindi / Telugu)
        │
        ▼
 Sentence-Transformers Embedding
 (all-MiniLM-L6-v2)
        │
        ▼
 ChromaDB Similarity Search
 (similarity_search_with_score)
        │
        ▼
 Top-5 Relevant Legal Chunks
 (with relevance filtering)
        │
        ▼
 Context Builder + Prompt Template
        │
        ▼
 Llama 3.2 via Ollama (streaming)
        │
        ▼
 Cited Legal Answer → Streamlit UI
```

---

## 📂 Folder Structure

```
law-ai-india/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── constitution.pdf      # Place legal PDFs here
│   ├── bns.pdf
│   ├── bnss.pdf
│   ├── bsa.pdf
│   ├── additional_laws/      # Extra acts
│   └── new_laws/             # Drop new PDFs here for auto-indexing
├── chroma_db/                # Persistent vector store (auto-created)
├── utils/
│   ├── __init__.py
│   ├── loader.py             # PDF loader + chunker + metadata extractor
│   ├── embedder.py           # HuggingFace + ChromaDB wrapper
│   ├── retriever.py          # Semantic retrieval
│   ├── llm.py                # Ollama streaming LLM
│   ├── ingestion.py          # Ingestion pipeline + folder monitor
│   ├── prompt_templates.py   # System & user prompt engineering
│   └── export.py             # PDF export utility
├── screenshots/
│   └── demo.png
└── assets/
    └── logo.png
```

---

## 📚 Supported Laws

| Act | Filename |
|---|---|
| Constitution of India | `constitution.pdf` |
| Bharatiya Nyaya Sanhita 2023 | `bns.pdf` |
| Bharatiya Nagarik Suraksha Sanhita 2023 | `bnss.pdf` |
| Bharatiya Sakshya Adhiniyam 2023 | `bsa.pdf` |
| Information Technology Act 2000 | `it_act.pdf` |
| Companies Act 2013 | `companies_act.pdf` |
| Income Tax Act 1961 | `income_tax.pdf` |
| GST Laws | `gst.pdf` |
| Consumer Protection Act 2019 | `consumer_protection.pdf` |
| Labour Laws | `labour.pdf` |
| Environment Protection Act | `environmental.pdf` |
| Banking Regulation Act | `banking.pdf` |
| Indian Contract Act 1872 | `contract.pdf` |
| Transfer of Property Act | `property.pdf` |
| Civil Procedure Code | `cpc.pdf` |
| Any future act | Drop in `/data/new_laws/` |

Filename → Act Name mapping is automatic. See `utils/loader.py` for the full map.

---

## 🔑 Why These Documents?

The initial corpus covers:
- **Constitutional law** (supreme law of the land)
- **New criminal codes** (BNS/BNSS/BSA — replaced IPC/CrPC/Evidence Act in 2024)
- **Commercial law** (Companies Act, Contract Act, GST, Income Tax)
- **Digital law** (IT Act, Cyber Laws)
- **Consumer & labour rights** (most-queried areas by citizens)

The system is designed to scale to all 1,500+ Central Acts on India Code.

---

## ✂️ Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~200 words per chunk
    chunk_overlap=200,    # preserves cross-boundary context
    separators=["\n\n", "\n", ".", " ", ""],
)
```

Each chunk stores:
```json
{
  "source": "/data/constitution.pdf",
  "page_number": 42,
  "act_name": "Constitution of India",
  "chapter": "IV",
  "section": "14"
}
```

---

## 🤖 Embedding Model

**`sentence-transformers/all-MiniLM-L6-v2`**

- 384-dimensional vectors
- Multilingual capability (supports Hindi/Telugu via Unicode)
- Fast CPU inference
- MIT licensed

Embeddings are persisted in ChromaDB; re-ingestion is skipped for already-indexed chunks (content-hash deduplication).

---

## 🔍 Retrieval Pipeline

1. Query is embedded using the same model
2. `similarity_search_with_score()` returns cosine distances
3. Chunks above the similarity threshold are filtered out
4. Top-5 results are returned with metadata
5. Results are shown in the UI expander with relevance %

---

## 💬 Prompt Engineering

**System prompt principles:**
- Answer ONLY from retrieved context (no hallucination)
- Respond in the user's language (EN/HI/TE)
- Cite Act + Section + Page in every response
- Clearly state when information is not found
- Educational disclaimer on every response

See `utils/prompt_templates.py` for full implementation.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed

### 1. Clone the repository

```bash
git clone https://github.com/yourname/law-ai-india.git
cd law-ai-india
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 5. Pull the Ollama model

```bash
ollama pull llama3.2
ollama serve          # keep this running in a separate terminal
```

### 6. Add legal PDFs

Place your PDFs in the `data/` directory:

```bash
cp /path/to/constitution.pdf data/
cp /path/to/bns.pdf data/
# etc.
```

### 7. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 8. Ingest documents

Click **"🔄 Ingest All PDFs in /data"** in the sidebar. This embeds all PDFs and stores them in ChromaDB. Only new PDFs are processed on subsequent runs.

---

## 📖 Usage

1. **Ask a question** in the chat box (English / Hindi / Telugu)
2. The system retrieves the 5 most relevant legal chunks
3. Llama 3.2 generates a cited answer based only on those chunks
4. Expand **"Legal Citations & Retrieved Context"** to see sources
5. Download the response as a PDF or export chat history

**Example queries:**

- *What are the Fundamental Rights under the Constitution?*
- *धारा 302 के अंतर्गत हत्या की सजा क्या है?*
- *IT చట్టం కింద సైబర్ నేరాలకు శిక్ష ఏమిటి?*

---

## 🔮 Future Improvements

- [ ] GPU-accelerated embeddings
- [ ] Multilingual embedding model (LaBSE) for better HI/TE support
- [ ] Full-text search fallback (BM25 hybrid retrieval)
- [ ] India Code API integration for real-time law updates
- [ ] Case law database (Supreme Court judgements)
- [ ] Section-level citation links
- [ ] WhatsApp / Telegram bot interface
- [ ] Role-based access (lawyer vs citizen mode)

---

## ⚠️ Legal Disclaimer

**LawAI India is for educational and research purposes only.**

This tool does not provide legal advice. The information generated is based on the documents indexed and may not reflect the most current legal position. Always consult a qualified legal professional for advice on specific legal matters.

---

## 📄 License

MIT License — see `LICENSE` file.

---

*Built with ❤️ for legal access in India.*
