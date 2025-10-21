# Fitsy Wardrobe AI Assistant

This project is a Streamlit-based AI assistant designed to help users manage their wardrobe and get outfit recommendations. It leverages large language models (LLMs) for fashion advice and vector databases (ChromaDB and in-memory FAISS) for efficient similarity search of clothing items.

## Project Structure

Here's an overview of the key files and directories in this project:







- **`get_data.py`**: This script is likely used for direct interaction with the ChromaDB. It provides functionality to connect to the database, list collections, and perform similarity searches, often used for debugging or data exploration outside the main Streamlit application.


- **`requirements.txt`**: This file lists all the Python dependencies required to run the project. It's essential for setting up the correct development environment by installing all necessary libraries.


## Technical Approaches

This project employs several key technologies and approaches to deliver its functionality:

- **Streamlit**: The entire user interface and interactive web application are built using [Streamlit](https://streamlit.io/). Streamlit allows for rapid development of data apps in Python, making it easy to create interactive components, display data, and manage application state.

- **ChromaDB**: For persistent storage of clothing item embeddings and their associated metadata, the project utilizes [ChromaDB](https://www.trychroma.com/). ChromaDB acts as a vector database, allowing for efficient storage and retrieval of high-dimensional vectors (embeddings) and their corresponding details. This ensures that the wardrobe items are saved and accessible across different sessions.

- **FAISS (Facebook AI Similarity Search)**: To enable fast and efficient similarity searches for outfit recommendations, an in-memory [FAISS](https://github.com/facebookresearch/faiss) index is employed. When the application starts and items are loaded from ChromaDB, a `faiss.IndexFlatIP` (Inner Product) index is created in memory. This index is used to quickly find clothing items whose embeddings are most similar to a given query, facilitating relevant outfit suggestions. While `faiss_metadata.json` exists, the active application currently reconstructs the FAISS index from ChromaDB data rather than loading from a file.

- **Groq (for LLM Inference)**: The core of the AI assistant's reasoning and recommendation generation is powered by Large Language Models (LLMs) served via [Groq](https://groq.com/). Specifically, `ChatGroq` and the `Groq` client are used to interact with high-performance LLMs (like `meta-llama/llama-4-scout-17b-16e-instruct` for summarization and `openai/gpt-oss-120b` for recommendations). Groq's fast inference capabilities ensure that outfit recommendations and chat responses are generated quickly.

- **Sentence Transformers**: Item embeddings, which are crucial for similarity searches in both FAISS and ChromaDB, are generated using the `SentenceTransformer('all-MiniLM-L6-v2')` model. This model converts textual descriptions of clothing items into dense vector representations.

## Setup and Running the Application

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd fitsy
    ```

2.  **Create a virtual environment (recommended):
    ```bash
    python -m venv fitsy_env
    fitsy_env\Scripts\activate  # On Windows
    source fitsy_env/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root and add your Groq API key:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```
    (Replace `your_groq_api_key_here` with your actual key.)

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app_alpha.py
    ```

This will launch the application in your web browser, typically at `http://localhost:8501`.
