# 📄 Resume Retrieval System using ChromaDB

## 📌 Project Title
Performance Comparison of Vector Databases and Retrieval Algorithms for Resume Search

---

## 📖 Overview

This project implements a semantic resume retrieval system using ChromaDB and multiple sentence embedding models. It evaluates different retrieval algorithms (BM25, Brute Force, and HNSW) on a large collection of resumes converted from PDF files.

The system measures retrieval latency, retrieval quality, and Top-K relevance across different embedding models.

This project is designed for academic research and benchmarking purposes.

---

## 🗂️ Folder Structure

RESUME_PARSER_CHROMADB/
│
├── chroma_db_bge/
├── chroma_db_minilm/
├── chroma_db_mpnet/
│
├── data/
│
├── bge_chromadb.py
├── minilm_chromadb.py
├── mpnet_chromadb.py
│
├── retrieval_benchmark.py
├── verify.py
│
├── performance_results.csv
├── comparative_summary.csv
│
├── requirements.txt
└── README.md


---

## 📊 Dataset Description

- Dataset Type: Resume PDF files  
- Source: Technical and educational profiles  
- Format: PDF → Text → Chunks  
- Storage: ChromaDB Vector Database  

### Processing Pipeline

PDF → Text Extraction → Chunking → Embedding → Vector Storage


---

## 🧠 Embedding Models Used

| Model | Dimension | Purpose |
|-------|-----------|----------|
| all-MiniLM-L6-v2 | 384 | Fast, lightweight |
| all-mpnet-base-v2 | 768 | High accuracy |
| BAAI/bge-base-en-v1.5 | 768 | High semantic quality |

---

## 🔍 Retrieval Algorithms

| Algorithm | Category | Description |
|-----------|----------|-------------|
| BM25 | Lexical | Keyword-based search |
| Brute Force | Exact | Full cosine similarity scan |
| HNSW | ANN | Approximate nearest neighbor |

---

## ⚙️ Evaluation Configuration

| Parameter | Value |
|-----------|--------|
| Test Queries | 20 |
| Query Type | Natural Language |
| Top-K Results | 5 |
| Similarity Metric | Cosine Similarity |
| Threads | 3 |

---

## 📈 Performance Metrics

The system evaluates:

- Average Response Time (ms)
- Retrieval Quality
- Ranking Accuracy
- Scalability

Results are saved to:
performance_results.csv
comparative_summary.csv

---

## 🚀 Installation

### 1. Create Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate


### 2. Install Dependencies

pip install -r requirements.txt


---

## ▶️ How to Run the Project

### Step 1: Prepare Dataset

Place all resume PDFs inside:

data/


---

### Step 2: Index Documents

Run the following scripts:

python minilm_chromadb.py
python mpnet_chromadb.py
python bge_chromadb.py


These scripts will:

- Read PDFs
- Generate embeddings
- Store vectors in ChromaDB

---

### Step 3: Verify Database

python verify.py


This confirms that all collections exist.

---

### Step 4: Run Benchmark

python retrieval_benchmark.py


This script will:

- Run 20 queries
- Perform BM25, Brute Force, and HNSW search
- Print Top-5 results
- Measure latency
- Save CSV reports

---

## 📄 Output Files

### performance_results.csv

Contains:

Retrieval Method, Embedding Model, Avg Latency, Quality


---

### comparative_summary.csv

Contains:

Configuration, Avg Latency, Accuracy Level


---

## 📊 Sample Output

==============================================
Performance Results (ChromaDB)
==============================================
Retrieval      Model          Latency (ms)      Quality
-----------------------------------------------------------------
BM25           N/A            39                Medium
Brute          MiniLM         91                High
HNSW           MiniLM         9                 High
-----------------------------------------------------------------

==============================================
Comparative Summary
==============================================

Configuration       Avg Latency (ms)    Accuracy Level
------------------------------------------------------------
BM25                41                  Medium
Brute Force         127                 High
HNSW (ChromaDB)     9                   High

---

## ✅ Key Findings

- BM25 is fast but lacks semantic understanding
- Brute force search is accurate but slow
- HNSW provides the best speed–accuracy balance
- MPNet and BGE produce higher retrieval quality
- MiniLM is suitable for low-resource environments

---

## 🧪 Technologies Used

- Python 3.10
- ChromaDB
- Sentence Transformers
- PyTorch
- Scikit-learn
- Rank-BM25
- NumPy
- PyPDF

---

## 📌 Applications

- Resume Search Engines
- HR Screening Systems
- Candidate Matching
- Talent Recommendation
- Academic Benchmarking

---

## 📚 Future Enhancements

- Web Dashboard
- Real-time Search API
- Hybrid Search
- Cloud Deployment
- Feedback-based Ranking

---

## 👨‍💻 Author

Developed by: P Venkata Rakesh 
Domain: Data Science 

---

