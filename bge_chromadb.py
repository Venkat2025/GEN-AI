import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ================= CONFIG =================

PDF_FOLDER = "./data/pdfs"
DB_PATH = "./chroma_db_bge"
COLLECTION_NAME = "bge"

CHUNK_SIZE = 500
BATCH = 500

# =========================================


print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# Persistent client
client = chromadb.Client(
    Settings(
        persist_directory=DB_PATH,
        is_persistent=True
    )
)


# Create / load collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================
# PDF PROCESSING
# =========================================

def read_pdf(path):

    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "

    return text


def chunk(text, size):

    return [text[i:i+size] for i in range(0, len(text), size)]


print("Reading PDFs...")

chunks = []


for f in os.listdir(PDF_FOLDER):

    if f.lower().endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, f)

        txt = read_pdf(pdf_path)

        chunks.extend(chunk(txt, CHUNK_SIZE))


print("Total chunks:", len(chunks))


# =========================================
# EMBEDDINGS
# =========================================

print("Encoding with BGE...")

embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True
)


# =========================================
# STORE
# =========================================

print("Storing to ChromaDB...")


for i in range(0, len(chunks), BATCH):

    docs = chunks[i:i+BATCH]
    embs = embeddings[i:i+BATCH]

    ids = [f"bge_{j}" for j in range(i, i+len(docs))]


    collection.add(
        documents=docs,
        embeddings=embs.tolist(),
        ids=ids
    )


    print("Batch", i//BATCH + 1, "done")


print(" BGE Indexed & Saved Successfully")
