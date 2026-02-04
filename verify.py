import chromadb
from chromadb.config import Settings

paths = [
    "./chroma_db_minilm",
    "./chroma_db_mpnet",
    "./chroma_db_bge"
]

for p in paths:

    print(f"\nChecking: {p}")

    client = chromadb.Client(
        Settings(
            persist_directory=p,
            is_persistent=True
        )
    )

    cols = client.list_collections()

    if not cols:
        print("❌ No collections found")
    else:
        for c in cols:
            print("✅ Found:", c.name)
