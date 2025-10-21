from huggingface_hub.inference._generated.types import document_question_answering
import chromadb

# Load your collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("wardrobe")

# Delete by document content using the correct where_document syntax
collection.delete(
    where_document={
        "$contains": "bottom dark blue straight-leg male fall casual plain denim solid relaxed"
    }
)

