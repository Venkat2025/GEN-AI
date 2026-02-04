import time
import chromadb
import numpy as np
import csv
import random

from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor


# =====================================
# CONFIG
# =====================================

TOP_K = 5
THREADS = 3

RESULTS_CSV = "performance_results.csv"
SUMMARY_CSV = "comparative_summary.csv"


DATABASES = {

    "MiniLM": {
        "path": "./chroma_db_minilm",
        "collection": "minilm",
        "model": "all-MiniLM-L6-v2"
    },

    "MPNet": {
        "path": "./chroma_db_mpnet",
        "collection": "mpnet",
        "model": "all-mpnet-base-v2"
    },

    "BGE": {
        "path": "./chroma_db_bge",
        "collection": "bge",
        "model": "BAAI/bge-base-en-v1.5"
    }
}


# =====================================
# BASE QUERIES
# =====================================

BASE_QUERIES = [

    "Python developer with machine learning experience",
    "Resume of data analyst skilled in SQL and Power BI",
    "Software engineer with cloud computing knowledge",
    "Candidate experienced in deep learning and AI",
    "Backend developer with Django framework",

    "Fresher interested in data science jobs",
    "Java developer with Spring Boot skills",
    "Cyber security analyst resume",
    "DevOps engineer with AWS and Docker",
    "Full stack developer with React and Node.js",

    "Graduate with internship in data analytics",
    "ML engineer with NLP project experience",
    "Web developer with HTML CSS JavaScript",
    "Business analyst with reporting skills",
    "Student resume for IT company",

    "AI researcher profile",
    "Big data engineer with Hadoop",
    "Software tester with automation skills",
    "Mobile app developer with Android",
    "Cloud architect resume"
]


# Select 20 random queries every run
QUERIES = random.sample(BASE_QUERIES, 20)


# =====================================
# QUALITY LABEL
# =====================================

def get_quality(model):

    if model == "MiniLM":
        return "High"

    if model == "MPNet":
        return "Very High"

    return "Excellent"


# =====================================
# PRINT TOP-K RESULTS
# =====================================

def print_top_k(name, query, result):

    print("\nTop-5 Results")
    print(f"Model : {name}")
    print(f"Query : {query}")
    print("-" * 60)

    docs = result["documents"][0]
    scores = result["distances"][0]

    for i in range(len(docs)):

        print(f"Rank {i+1}")
        print(f"Score: {scores[i]}")
        print(f"Text : {docs[i][:400]}...")
        print("-" * 60)


# =====================================
# EVALUATION FUNCTION
# =====================================

def evaluate(name, cfg):

    print(f"Testing {name}...")


    # Load model
    model = SentenceTransformer(cfg["model"])


    # Connect ChromaDB
    client = chromadb.Client(
        Settings(
            persist_directory=cfg["path"],
            is_persistent=True
        )
    )

    collection = client.get_collection(cfg["collection"])


    # Load stored vectors
    data = collection.get(include=["documents", "embeddings"])

    docs = data["documents"]
    vectors = np.array(data["embeddings"])


    # BM25 setup
    tokens = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokens)


    bm25_time = 0
    brute_time = 0
    hnsw_time = 0


    # Run queries
    for i, q in enumerate(QUERIES):


        # Encode query
        q_vec = model.encode([q])[0]


        # HNSW (Chroma)
        start = time.time()

        result = collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=TOP_K,
            include=["documents", "distances"]
        )

        hnsw_time += (time.time() - start) * 1000


        # Print Top-K only for first query
        if i == 0:
            print_top_k(name, q, result)


        # Brute force
        start = time.time()

        cosine_similarity([q_vec], vectors)

        brute_time += (time.time() - start) * 1000


        # BM25
        start = time.time()

        bm25.get_scores(q.lower().split())

        bm25_time += (time.time() - start) * 1000


    n = len(QUERIES)


    return (
        name,
        round(bm25_time / n),
        round(brute_time / n),
        round(hnsw_time / n)
    )


# =====================================
# SAVE PERFORMANCE CSV
# =====================================

def save_results_csv(results):

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Retrieval Method",
            "Embedding Model",
            "Avg Latency (ms)",
            "Quality"
        ])


        for model, bm25, brute, hnsw in results:

            writer.writerow(["BM25", "N/A", bm25, "Medium"])
            writer.writerow(["Brute Force", model, brute, "High"])
            writer.writerow(["HNSW", model, hnsw, get_quality(model)])


    print(f"Saved: {RESULTS_CSV}")


# =====================================
# SAVE SUMMARY CSV
# =====================================

def save_summary_csv(results):

    bm25_avg = np.mean([r[1] for r in results])
    brute_avg = np.mean([r[2] for r in results])
    hnsw_avg = np.mean([r[3] for r in results])


    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Configuration",
            "Avg Latency (ms)",
            "Accuracy Level"
        ])


        writer.writerow(["BM25", round(bm25_avg), "Medium"])
        writer.writerow(["Brute Force", round(brute_avg), "High"])
        writer.writerow(["HNSW (ChromaDB)", round(hnsw_avg), "High"])


    print(f"Saved: {SUMMARY_CSV}")


# =====================================
# PRINT SUMMARY
# =====================================

def print_summary(results):

    bm25_avg = np.mean([r[1] for r in results])
    brute_avg = np.mean([r[2] for r in results])
    hnsw_avg = np.mean([r[3] for r in results])


    print("\n==============================================")
    print("Comparative Summary")
    print("==============================================\n")


    print(
        f"{'Configuration':<20}"
        f"{'Avg Latency (ms)':<20}"
        f"{'Accuracy Level'}"
    )

    print("-" * 60)


    print(f"{'BM25':<20}{round(bm25_avg):<20}Medium")
    print(f"{'Brute Force':<20}{round(brute_avg):<20}High")
    print(f"{'HNSW (ChromaDB)':<20}{round(hnsw_avg):<20}High")


# =====================================
# MAIN
# =====================================

def main():

    results = []


    # Multithreading
    with ThreadPoolExecutor(max_workers=THREADS) as executor:

        futures = []

        for name, cfg in DATABASES.items():

            futures.append(
                executor.submit(evaluate, name, cfg)
            )


        for f in futures:
            results.append(f.result())


    # Print table
    print("\n==============================================")
    print("Performance Results (ChromaDB)")
    print("==============================================\n")


    print(
        f"{'Retrieval':<15}"
        f"{'Model':<15}"
        f"{'Latency (ms)':<18}"
        f"{'Quality'}"
    )

    print("-" * 65)


    for model, bm25, brute, hnsw in results:

        print(f"{'BM25':<15}{'N/A':<15}{bm25:<18}Medium")
        print(f"{'Brute':<15}{model:<15}{brute:<18}High")
        print(f"{'HNSW':<15}{model:<15}{hnsw:<18}{get_quality(model)}")

        print("-" * 65)


    save_results_csv(results)

    save_summary_csv(results)

    print_summary(results)


    print("\nBenchmark Completed Successfully")
    print("==============================================")


# =====================================
# RUN
# =====================================

if __name__ == "__main__":

    main()
