import chromadb

# Connect to the existing Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# List available collections
collections = chroma_client.list_collections()
print(f"Found {len(collections)} collections\n")

for col in collections:
    print(f"--- Collection: {col.name} ---")

# Ask user which collection to query
col_name = input("Enter the collection name you want to search: ").strip()
collection = chroma_client.get_collection(col_name)

# Ask user for a query
query = input("Enter your query: ").strip()

# Perform similarity search (FAISS under the hood)
results = collection.query(
    query_texts=[query],
    n_results=30,   # number of closest matches to return
    include=["documents", "metadatas","data"]
)

# Print results
print("\n--- Search Results ---")
for i in range(len(results["documents"][0])):
    print(f"Result {i+1}:")
    print(f"Document: {results['documents'][0][i]}")
    print(f"Metadata: {results['metadatas'][0][i]}")
    print("-----")
