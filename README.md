# arXiv Materials Informatics RAG

A Retrieval-Augmented Generation (RAG) system specialized in Materials Informatics literature from arXiv. The system retrieves relevant excerpts from academic papers and generates grounded answers with provenance citations (`[paper | chunk]`).

---

## 🔑 Prerequisites & API Tokens

Before running the application or evaluation pipelines, you need to configure your environment variables:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and provide your API keys:
   ```env
   # Hugging Face Access Token (required for embedding / HF inference models)
   HF_TOKEN=your_huggingface_token_here

   # OpenAI API Key (required for running evaluation scripts with gpt-4o-mini)
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   > **Note:**
   > - You can obtain a free Hugging Face token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   > - You can obtain an OpenAI API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshitakrishna18/rag-material-informatics.git
   cd rag-material-informatics
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Pipeline Overview

The project is structured into sequential steps:

### 1. Data Ingestion (`scripts/1_download_arxiv.py`)
Queries the arXiv API for papers in materials informatics (`cat:cond-mat.mtrl-sci`) and downloads PDFs into `data/pdfs/` with rate-limit handling.

```bash
python scripts/1_download_arxiv.py
```

### 2. PDF Parsing (`scripts/2_pdf_to_text.py`)
Extracts text from downloaded PDFs page-by-page using `pypdf` and saves clean text files into `data/text/`.

```bash
python scripts/2_pdf_to_text.py
```

### 3. Chunking & Indexing (`scripts/3_build_index.py`)
- Chunks text into sliding windows (2,200 characters with 300-character overlap).
- Encodes chunks into dense embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Normalizes vectors and constructs an inner-product FAISS vector index (`IndexFlatIP`).

```bash
python scripts/3_build_index.py
```

---

## 🖥️ Running the Interactive App

Launch the Streamlit web interface to query the indexed literature:

```bash
streamlit run app_streamlit.py
```

- Adjust the top-$k$ retrieval parameter from the sidebar.
- Inspect retrieved paper chunks and relevance scores.
- View answers synthesized with source citations.

---

## 📊 Evaluation & Metrics

The project includes comparative evaluation against baselines:

1. **Generate Evaluation Answers (`scripts/4_generate_eval.py`):**
   Runs test questions across three regimes:
   - **Baseline:** Ungrounded, zero-shot LLM (no retrieval context).
   - **Ablation:** Sparse keyword retrieval via BM25 (`BM25Okapi`).
   - **RAG System:** Dense vector retrieval via FAISS + `all-MiniLM-L6-v2`.

   ```bash
   python scripts/4_generate_eval.py
   ```

2. **Calculate Evaluation Metrics (`calc_final_metrics.py`):**
   Computes BLEU-4 and ROUGE-1 / ROUGE-L scores against curated reference answers:

   ```bash
   python calc_final_metrics.py
   ```

---

## 📁 Repository Structure

```text
├── app_streamlit.py        # Streamlit interactive RAG demo
├── calc_final_metrics.py   # BLEU & ROUGE evaluation calculation
├── eval_dataset.json       # Benchmark questions & ground truth references
├── eval_results_mock.json  # Output answers across baseline, ablation & RAG
├── requirements.txt        # Python package dependencies
├── scripts/
│   ├── 1_download_arxiv.py # Download papers from arXiv API
│   ├── 2_pdf_to_text.py    # Extract text from PDFs
│   ├── 3_build_index.py    # Chunking & FAISS index creation
│   └── 4_generate_eval.py  # Run multi-mode evaluation generation
├── .env.example            # Template for environment variables / tokens
└── README.md               # Project documentation
```
