import streamlit as st
import json
import os
import time
import requests
from datetime import datetime
import chromadb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from langchain_groq import ChatGroq
from groq import Groq
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser as strout
from langchain.memory import ConversationSummaryMemory
import re
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Fitsy Wardrobe AI Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .outfit-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .item-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .chat-message {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        border: 2px solid #667eea;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #667eea;
        border-radius: 8px;
    }
    
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .image-item {
        text-align: center;
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Consistent image frame styling */
    .uniform-image-container {
        width: 100%;
        max-width: 300px;
        height: 400px;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 10px auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        box-sizing: border-box;
    }
    
    .uniform-image-container img {
        max-width: 100%;
        max-height: 300px;
        object-fit: contain;
        border-radius: 8px;
        display: block;
    }
    
    .uniform-image-container .item-details {
        margin-top: 10px;
        text-align: center;
        width: 100%;
        font-size: 0.9em;
        line-height: 1.4;
    }
    
    /* Responsive grid adjustments */
    .stColumns > div {
        display: flex;
        justify-content: center;
    }
    
    /* Ensure consistent spacing between image containers */
    .uniform-image-container {
        margin: 15px auto;
    }
    
    /* Responsive adjustments for different screen sizes */
    @media (max-width: 768px) {
        .uniform-image-container {
            max-width: 250px;
            height: 350px;
        }
    }
    
    /* Hover effects for better user experience */
    .uniform-image-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    /* Ensure images maintain aspect ratio */
    .uniform-image-container img {
        aspect-ratio: auto;
        object-position: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_items' not in st.session_state:
        st.session_state.processed_items = {}
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'item_embeddings' not in st.session_state:
        st.session_state.item_embeddings = {}
    if 'eye_color' not in st.session_state:
        st.session_state.eye_color = "Not specified"
    if 'body_type' not in st.session_state:
        st.session_state.body_type = "Not specified"
    if 'ethnicity' not in st.session_state:
        st.session_state.ethnicity = "Not specified"

# Load environment variables
@st.cache_resource
def load_env_and_models():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.stop()
    
    # Initialize models
    summary_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    recommendation_model = "openai/gpt-oss-120b"
    
    llm = ChatGroq(model=recommendation_model, temperature=0.8)
    client = Groq(api_key=api_key)
    
    # Initialize conversation summary memory for tracking outfit recommendations
    recommendation_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="history",
        return_messages=True,
        max_token_limit=1000  # Limit summary length
    )
    
    # Load sentence transformer for FAISS
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return api_key, llm, client, summary_model, embedding_model, recommendation_memory

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="wardrobe")
    return chroma_client, collection

# FAISS operations
def create_faiss_index(items, embedding_model):
    """Create FAISS index from wardrobe items"""
    if not items:
        return None, {}
    
    # Create embeddings for all items
    item_texts = []
    item_names = []
    
    for item_name, item_details in items.items():
        if isinstance(item_details, dict):
            # Create text representation of item
            text = f"{item_details.get('category', '')} {item_details.get('color', '')} {item_details.get('style', '')} {item_details.get('gender', '')} {item_details.get('season', '')} {item_details.get('fashion_type', '')} {item_details.get('design', '')} {item_details.get('fabric', '')} {item_details.get('pattern', '')}"
            item_texts.append(text)
            item_names.append(item_name)
    
    if not item_texts:
        return None, {}
    
    # Generate embeddings
    embeddings = embedding_model.encode(item_texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Create mapping
    id_to_name = {i: name for i, name in enumerate(item_names)}
    
    return index, id_to_name

def search_similar_items(query, faiss_index, id_to_name, embedding_model, items, top_k=10):
    """Search for similar items using FAISS"""
    if faiss_index is None:
        return []
    
    # Encode query
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    
    # Get results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx in id_to_name:
            item_name = id_to_name[idx]
            if item_name in items:
                results.append({
                    'name': item_name,
                    'score': float(score),
                    'details': items[item_name]
                })
    
    return results

# Utility functions
def validate_url(url):
    """Quick validation of URL accessibility"""
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except:
        return False

def create_uniform_image_display(url, name, details, fallback_text="Image unavailable"):
    """Create HTML for uniform image display with consistent dimensions"""
    try:
        if url:
            # Ensure consistent image dimensions
            processed_url = ensure_image_dimensions(url)
            return f"""
            <div class="uniform-image-container">
                <img src="{processed_url}" alt="{name}" style="max-width: 100%; max-height: 300px; object-fit: contain;">
                <div class="item-details">
                    <strong>{name}</strong><br>
                    Category: {details.get('category', 'N/A')}<br>
                    Color: {details.get('color', 'N/A')}<br>
                    Style: {details.get('style', 'N/A')}
                </div>
            </div>
            """
        else:
            # No URL provided, show fallback
            return f"""
            <div class="uniform-image-container">
                <div class="item-details">
                    <strong>{name}</strong><br>
                    Category: {details.get('category', 'N/A')}<br>
                    Color: {details.get('color', 'N/A')}<br>
                    Style: {details.get('style', 'N/A')}<br>
                    <em>{fallback_text}</em>
                </div>
            </div>
            """
    except Exception as e:
        # Error occurred, show fallback
        return f"""
        <div class="uniform-image-container">
            <div class="item-details">
                <strong>{name}</strong><br>
                Category: {details.get('category', 'N/A')}<br>
                Color: {details.get('color', 'N/A')}<br>
                Style: {details.get('style', 'N/A')}<br>
                <em>{fallback_text}</em>
            </div>
        </div>
        """

def ensure_image_dimensions(url, target_width=300, target_height=300):
    """Ensure image has consistent dimensions by preprocessing if needed"""
    try:
        # For now, we'll use CSS to handle the sizing
        # In a production environment, you might want to actually resize images
        return url
    except Exception as e:
        st.warning(f"Could not process image dimensions: {e}")
        return url

def load_existing_items(collection):
    """Load existing items from ChromaDB"""
    try:
        existing_items = collection.get()
        items = {}
        
        if existing_items["ids"]:
            for i, item_id in enumerate(existing_items["ids"]):
                metadata = existing_items["metadatas"][i]
                items[item_id] = metadata
        
        return items
    except Exception as e:
        st.error(f"Error loading existing items: {e}")
        return {}

@st.cache_data
def load_and_display_image(url, caption, item_name, item_details):
    """Load and display an image with consistent frame dimensions"""
    try:
        # Use the utility function for consistent display
        html_content = create_uniform_image_display(url, item_name, item_details)
        st.markdown(html_content, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.error(f"Error loading image for {item_name}: {e}")
        # Fallback to text display if image fails
        html_content = create_uniform_image_display("", item_name, item_details, "Image unavailable")
        st.markdown(html_content, unsafe_allow_html=True)
        return False

def display_outfit_recommendation(rec, items, index):
    """Display a single outfit recommendation"""
    st.markdown(f"""
    <div class="outfit-card">
        <h3>🎯 Outfit {index}</h3>
        <p><strong>Recommendation:</strong> {rec.get('recommendation', '')}</p>
        <p><strong>Reason:</strong> {rec.get('reason', '')}</p>
        <p><strong>Missing Items:</strong> {', '.join(rec.get('missing_items', []))}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recommended images
    image_names = rec.get('image_names', [])
    if image_names:
        st.subheader("👕 Recommended Items")
        
        # Create columns for images
        cols = st.columns(min(len(image_names), 4))
        
        for i, name in enumerate(image_names):
            with cols[i % len(cols)]:
                if name in items:
                    item_details = items[name]
                    url = item_details.get('url', '')
                    
                    if url:
                        # Use the cached image loading function
                        load_and_display_image(url, name, name, item_details)
                else:
                    st.warning(f"Item '{name}' not found in database")

# Configuration
BATCH_SIZE = 3  # Process 3 images at a time (within the 5 image limit)
MAX_RETRIES = 3  # Maximum retry attempts for failed batches

# Ensure batch size doesn't exceed API limit
if BATCH_SIZE > 5:
    st.warning("⚠️ Batch size reduced to 5 (API limit)")
    BATCH_SIZE = 5

def process_new_urls(urls, client, summary_model, collection):
    """Process new URLs and add to database"""
    if not urls:
        return {}
    
    # Limit to 5 images per batch (Groq API limit)
    MAX_IMAGES_PER_BATCH = 5
    if len(urls) > MAX_IMAGES_PER_BATCH:
        st.warning(f"⚠️ Processing {len(urls)} images in batches of {MAX_IMAGES_PER_BATCH} (API limit)")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process URLs in batches of 5
    all_items = {}
    total_batches = (len(urls) + MAX_IMAGES_PER_BATCH - 1) // MAX_IMAGES_PER_BATCH
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * MAX_IMAGES_PER_BATCH
        end_idx = min(start_idx + MAX_IMAGES_PER_BATCH, len(urls))
        batch_urls = urls[start_idx:end_idx]
        
        st.write(f"🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_urls)} images)")
        
        # Create prompt for processing this batch
        urls_str = json.dumps(batch_urls)
        
        prompt = f"""You are a fashion wardrobe analyzer. Your task is to analyze the provided clothing images and create descriptive names for each item.
Task:
Your task is that you should provie the data in that way so that when this passed to a new model to parse this it should not get 
confused in gender or in any other way.
- Below are the instructions for the task:
INSTRUCTIONS:
1. Examine each image carefully and identify what type of clothing item it is.
2. Create a descriptive name for each item in format: "category_color_style_gender" (e.g., "top_white_oxford", "bottom_black_jeans").
3. Categorize each item as one of: top, bottom, footwear, accessory, or outerwear.
4. Return a single JSON object where each key is your created item name and value contains the original URL and attributes.

EXACT URLS TO USE (DO NOT MODIFY THESE):
{urls_str}

RESPONSE FORMAT (strict JSON, no extra text):
{{
  "item_name1": {{
    "url": "EXACT_URL_FROM_LIST_ABOVE",
    "category": "top/bottom/footwear/accessory/outerwear",
    "color": "descriptive color",
    "genre":"pants/shirt/jacket/etc.",
    "style": "brief style description",
    "gender": "male/female",
    "season": "spring/summer/fall/winter",
    "fashion_type": "casual/formal/sporty/etc.",
    "design": "brief design description",
    "fabric": "brief fabric description",       
    "pattern": "brief pattern description",
    "fit": "brief fit description",
    "occasion": "brief occasion description"
  }}
}}

CRITICAL REQUIREMENTS:
1. Response must be ONLY the JSON object - no markdown, no explanations, no extra text.
2. Ensure the JSON is properly formatted and valid.
3. Start your response with {{ and end with }} - nothing else.
4. Create unique, descriptive names that clearly identify each item.
5. Copy-paste the exact URLs from the list above - DO NOT create example URLs.
6. Analyze each image in the order provided and match URLs accordingly.
7. Be specific and detailed in your analysis - don't use generic descriptions.
8.The Response should be strictly in JSON format and nothing else
9.Moreover if there is an image in which there is a complete suit or it contains mutltiple you should extract multiple items from it 
like if my suit contains all the data like accessories,top,bottom,footwear,watch,etc. you should extract all the items from it and the json should contain all the items from it and it should be reprsented as a complete outfit  if it contain top,bttom.
and missing items like footwear should be included from the wardrobe if it suits else show it in missing items array.
like this example:
{{
  "item_name1": {{
    "url": "EXACT_URL_FROM_LIST_ABOVE",
    "category": "top,bottom,footwear,accessory,outerwear",
    "genre":"pants/shirt/jacket/etc.",
    "color_top": "descriptive color",
    "color_bottom": "descriptive color",
    "color_footwear": "descriptive color",
    "color_accessory": "descriptive color",
    "color_outerwear": "descriptive color",
    "style_top": "brief style description",
    "style_bottom": "brief style description",
    "style_footwear": "brief style description",
    "style_accessory": "brief style description",
    "style_outerwear": "brief style description",
    "season_top": "spring/summer/fall/winter",
    "season_bottom": "spring/summer/fall/winter",
    "season_footwear": "spring/summer/fall/winter",
    "season_accessory": "spring/summer/fall/winter",
    "season_outerwear": "spring/summer/fall/winter",
    "fashion_type_top": "casual/formal/sporty/etc.",
    "fashion_type_bottom": "casual/formal/sporty/etc.",
    "design_top": "brief design description",
    "design_bottom": "brief style description",
    "design_footwear": "brief style description",
    "design_accessory": "brief style description",
    "design_outerwear": "brief style description",
    "fabric_top": "brief fabric description",
    "fabric_bottom": "brief fabric description",
    "fabric_footwear": "brief fabric description",
    "fabric_accessory": "brief fabric description",
    "fabric_outerwear": "brief fabric description",
    "pattern_top": "brief pattern description",
    "pattern_bottom": "brief pattern description",
    "pattern_footwear": "brief pattern description",
    "pattern_accessory": "brief pattern description",
    "pattern_outerwear": "brief pattern description",
  }}
}}

"""

        try:
            status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}...")
            progress_bar.progress((batch_idx + 1) / total_batches)
            
            # Create messages for this batch
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in batch_urls
                    ]
                }
            ]
            
            # Process with Groq
            summary = client.chat.completions.create(
                model=summary_model,
                messages=messages,
                max_tokens=2048,
                timeout=60
            )
            
            response = summary.choices[0].message.content.strip()
            
            # Parse JSON
            batch_items = json.loads(response)
            
            # Add to ChromaDB
            for item_name, item_details in batch_items.items():
                # Create document for embedding
                document = f"{item_details.get('category', '')} {item_details.get('color', '')} {item_details.get('style', '')} {item_details.get('gender', '')} {item_details.get('season', '')} {item_details.get('fashion_type', '')} {item_details.get('design', '')} {item_details.get('fabric', '')} {item_details.get('pattern', '')} {item_details.get('fit', '')} {item_details.get('occasion', '')}"
                
                # Add to collection
                collection.add(
                    ids=[item_name],
                    documents=[document],
                    metadatas=[item_details]
                )
            
            # Merge batch results
            all_items.update(batch_items)
            st.success(f"✅ Batch {batch_idx + 1} processed successfully ({len(batch_items)} items)")
            
        except Exception as e:
            st.error(f"❌ Error processing batch {batch_idx + 1}: {e}")
            st.error(f"Batch URLs: {batch_urls}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text("✅ All batches processed!")
    
    return all_items

def main():
    # Initialize
    init_session_state()
    
    # Load models and initialize database
    api_key, llm, client, summary_model, embedding_model, recommendation_memory = load_env_and_models()
    chroma_client, collection = init_chromadb()
    
    # Header
    st.markdown('<h1 class="main-header">👗 Fashion Wardrobe AI Assistant</h1>', unsafe_allow_html=True)
    
    # Load existing items
    existing_items = load_existing_items(collection)
    
    # Create FAISS index
    if existing_items and st.session_state.faiss_index is None:
        st.session_state.faiss_index, st.session_state.item_embeddings = create_faiss_index(existing_items, embedding_model)
        st.session_state.embedding_model = embedding_model
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Database stats
        st.markdown("### 📊 Database Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(existing_items)}</h3>
                <p>Total Items</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            categories = {}
            for item in existing_items.values():
                if isinstance(item, dict):
                    cat = item.get('category', 'Unknown')
                    categories[cat] = categories.get(cat, 0) + 1
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(categories)}</h3>
                <p>Categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add new items section
        st.markdown("### ➕ Add New Items")
        
        new_urls_text = st.text_area(
            "Enter image URLs (one per line):",
            height=150,
            placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg"
        )
        
        if st.button("🚀 Process New Items", use_container_width=True):
            if new_urls_text.strip():
                urls = [url.strip() for url in new_urls_text.split('\n') if url.strip()]
                
                # Validate URLs
                valid_urls = [url for url in urls if validate_url(url)]
                
                if valid_urls:
                    with st.spinner("Processing images..."):
                        new_items = process_new_urls(valid_urls, client, summary_model, collection)
                        
                    if new_items:
                        st.success(f"✅ Added {len(new_items)} new items!")
                        existing_items.update(new_items)
                        
                        # Update FAISS index
                        st.session_state.faiss_index, st.session_state.item_embeddings = create_faiss_index(existing_items, embedding_model)
                        
                        st.rerun()
                else:
                    st.error("❌ No valid URLs found!")

        # User Preferences Section
        st.markdown("### 👤 Your Preferences")
        with st.expander("Configure your personal preferences", expanded=False):
            eye_color_options = ["Not specified", "Brown", "Blue", "Green", "Hazel", "Gray", "Amber", "Black"]
            st.session_state.eye_color = st.selectbox(
                "Eye Color:",
                eye_color_options,
                index=eye_color_options.index(st.session_state.eye_color) if st.session_state.eye_color in eye_color_options else 0,
                key="eye_color_select"
            )
            
            body_type_options = ["Not specified", "Triangle", "Inverted Triangle", "Rectangle", "Trapezoid", "Round", "Not sure"]
            st.session_state.body_type = st.selectbox(
                "Body Type:",
                body_type_options,
                index=body_type_options.index(st.session_state.body_type) if st.session_state.body_type in body_type_options else 0,
                key="body_type_select"
            )
            
            ethnicity_options = ["Not specified", "Asian", "Black", "Caucasian", "Hispanic", "Middle Eastern", "Native American", "South Asian", "Other"]
            st.session_state.ethnicity = st.selectbox(
                "Ethnicity:",
                ethnicity_options,
                index=ethnicity_options.index(st.session_state.ethnicity) if st.session_state.ethnicity in ethnicity_options else 0,
                key="ethnicity_select"
            )
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🎯 Outfit Recommendations", "💬 Chat Assistant"])
    
    # Tab 1: Recommendations
    with tab1:
        st.markdown('<div class="section-header">🎯 Get Outfit Recommendations</div>', unsafe_allow_html=True)
        
        if not existing_items:
            st.warning("⚠️ No items in database. Please add some items first using the sidebar.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                prompt = st.text_input(
                    "🔍 Describe your desired outfit:",
                    placeholder="e.g., formal party, casual weekend, business meeting"
                )
            
            with col2:
                num_recommendations = st.selectbox(
                    "Number of recommendations:",
                    [1, 2, 3, 4, 5,6,7,8,9,10],
                    index=2
                )
            
            if st.button("✨ Get Recommendations", use_container_width=True, type="primary"):
                if prompt:
                    with st.spinner("Creating perfect outfits for you..."):
                        # Use FAISS to find relevant items
                        if st.session_state.faiss_index is not None:
                            relevant_results = search_similar_items(
                                prompt, 
                                st.session_state.faiss_index, 
                                st.session_state.item_embeddings, 
                                embedding_model, 
                                existing_items,
                                top_k=30
                            )
                            relevant_items = {r['name']: r['details'] for r in relevant_results}
                        else:
                            # Fallback to ChromaDB query
                            query_results = collection.query(
                                query_texts=[prompt],
                                n_results=30
                            )
                            
                            relevant_items = {}
                            if query_results["ids"][0]:
                                for i in range(len(query_results["ids"][0])):
                                    item_name = query_results["ids"][0][i]
                                    metadata = query_results["metadatas"][0][i]
                                    relevant_items[item_name] = metadata
                        
                        if relevant_items:
                            # Create a container to hold recommendations
                            recommendations_container = st.container()
                            
                            with recommendations_container:
                                st.write("🎨 Generating outfit recommendations...")
                                
                                # Get conversation summary from memory to avoid duplicate recommendations
                                conversation_summary = recommendation_memory.moving_summary_buffer if hasattr(recommendation_memory, 'moving_summary_buffer') else ""
                                if not conversation_summary:
                                    # Try alternative method to get summary
                                    try:
                                        conversation_summary = recommendation_memory.load_memory_variables({})["history"]
                                    except:
                                        conversation_summary = ""
                                
                                history = conversation_summary if conversation_summary else "No previous recommendations yet."
                                
                                # Create recommendation prompt
                                recommendation_prompt = ChatPromptTemplate.from_messages([
                                    {"role": "system", "content": """You are a professional fashion stylist. Based on the wardrobe items provided, create {num_recommendations} complete outfit recommendations for: "{prompt}"

Available items: {context}
Additional user preferences:
- Eye Color: {eye_color}
- Body Type: {body_type}
- Ethnicity: {ethnicity}
Those preferences are very important and you should use them to generate the recommendations.
so that the recommendations are more personalized and relevant to the user.
Task:
Your task is that you should parse the data which will be used to provide the data to the user according to his prompt.
If the user asks for a formal outfit you should provide a formal outfit and if the user asks for a casual outfit you should provide a casual outfit and so on.
- Below are the instructions for the task:
IMPORTANT: You must respond with ONLY a valid JSON array. No explanations, no markdown, no extra text.
**Previous Recommendations History**:
{history}

CRITICAL: You must analyze the previous recommendations history above and ensure you NEVER repeat:
1. The exact same outfit combinations
2. The same specific clothing items that were already recommended
3. Similar color schemes or patterns that were already suggested
4.Just look at the clothes name and the prompt if the prompt is same(in the history) you should suggest those clothes which were not in the history before for the same prompt.
5.Instead of Full prompt Just look at the clothes name and the prompt if the prompt is same(in the history) you should suggest those clothes which were not in the history before for the same prompt.
6.When all the possible recommendations them you can start giving mixed response simple
If you see that certain items or combinations were already recommended, you MUST choose different items or create different combinations. This ensures the user gets diverse, non-repetitive outfit suggestions.

**Important:**
Return your response as a JSON array where each recommendation has:
- recommendation: Brief description of the complete outfit
- reason: Why this outfit works for the occasion 
- image_names: Array of item names that make up this outfit complete 
- You should generate complete outfits and not just a part of it.It should be a complete outfit containing all the items (top,bottom,footwear are necessary items you should include them always accesoires are optional means if they look good together you can include them) that are present in the available wardrobe.
and those items should look good together and
-Only Include missing items in the missing_items array if there is no relevant thing that can be used to complete the outfit.
- Be 100 percent specific in genders also specify in the recommendations.
-There should be no mixup of genders female data should be included in the female outfits and mens data should be included in the mens outfits.
Expected JSON format (copy this exactly and fill in your recommendations):
[
  {{
    "recommendation": "Complete outfit description and also tell why this outfit suits the user according to the eye color, body type, ethnicity.",
    "reason": "Why this works",
    "image_names": ["item1", "item2", "item3"](must contain top,bottom,footwear),
    "missing_items": ["missing_item1", "missing_item2"]
  }}
]

CRITICAL: 
1. Start your response with [ and end with ]
2. Use only items that exist in the available wardrobe
3. Be creative but practical with combinations
4. Consider color coordination, style matching, and occasion appropriateness
5. NO markdown formatting, NO explanations, ONLY the JSON array"""}
                                ])
                                
                                chain = recommendation_prompt | llm | strout()
                                
                                try:
                                    st.write("🤖 Invoking LLM with prompt...")
                                    st.write("Prompt variables:")
                                    st.json({
                                        "prompt": prompt,
                                        "context_length": len(json.dumps(relevant_items, indent=2)),
                                        "num_recommendations": num_recommendations
                                    })
                                    
                                    recommendation_str = chain.invoke({
                                        "prompt": prompt,
                                        "context": json.dumps(relevant_items, indent=2),
                                        "num_recommendations": num_recommendations,
                                        "history": history,
                                        "eye_color": st.session_state.eye_color,
                                        "body_type": st.session_state.body_type,
                                        "ethnicity": st.session_state.ethnicity
                                    })
                                    # Save to memory to track recommendations and avoid duplicates
                                    recommendation_memory.save_context(
                                        {"input": f"User requested: {prompt}"}, 
                                        {"output": f"Generated recommendations: {recommendation_str}"}
                                    )
                                    st.write("✅ LLM response received")
                                    st.write(f"Response length: {len(recommendation_str)} characters")
                                    
                                    # Debug: Show what the LLM returned
                                    st.write("🔍 Raw LLM response:")
                                    st.code(recommendation_str, language="text")
                                    
                                    # Check if response is empty or whitespace
                                    if not recommendation_str or recommendation_str.strip() == "":
                                        st.error("❌ LLM returned empty response. Please try again.")
                                        return
                                    
                                    # Try to clean the response if it contains markdown or extra text
                                    cleaned_response = recommendation_str.strip()
                                    
                                    # Remove markdown code blocks if present
                                    if cleaned_response.startswith("```json"):
                                        cleaned_response = cleaned_response[7:]
                                    if cleaned_response.endswith("```"):
                                        cleaned_response = cleaned_response[:-3]
                                    
                                    # Remove any leading/trailing whitespace
                                    cleaned_response = cleaned_response.strip()
                                    
                                    st.write("🧹 Cleaned response:")
                                    st.code(cleaned_response, language="json")
                                    
                                    # Parse recommendations
                                    try:
                                        recommendations = json.loads(cleaned_response)
                                    except json.JSONDecodeError as json_error:
                                        st.error(f"❌ Failed to parse JSON response: {json_error}")
                                        st.error("Raw response was:")
                                        st.code(cleaned_response, language="text")
                                        st.error("This usually means the AI didn't return valid JSON. Please try again.")
                                        return
                                    
                                    # Validate the parsed recommendations
                                    if not isinstance(recommendations, list):
                                        st.error("❌ Response is not a list. Expected a JSON array of recommendations.")
                                        st.error("Raw response was:")
                                        st.code(cleaned_response, language="text")
                                        return
                                    
                                    if len(recommendations) == 0:
                                        st.error("❌ No recommendations generated. Please try again.")
                                        return
                                    
                                    # Validate each recommendation
                                    valid_recommendations = []
                                    for i, rec in enumerate(recommendations):
                                        if not isinstance(rec, dict):
                                            st.warning(f"⚠️ Recommendation {i+1} is not a valid object, skipping...")
                                            continue
                                        
                                        required_fields = ['recommendation', 'reason', 'image_names', 'missing_items']
                                        missing_fields = [field for field in required_fields if field not in rec]
                                        
                                        if missing_fields:
                                            st.warning(f"⚠️ Recommendation {i+1} missing fields: {missing_fields}, skipping...")
                                            continue
                                        
                                        valid_recommendations.append(rec)
                                    
                                    if not valid_recommendations:
                                        st.error("❌ No valid recommendations found. Please try again.")
                                        return
                                    
                                    # Display valid recommendations
                                    st.write(f"🎉 Generated {len(valid_recommendations)} outfit recommendations!")
                                    
                                    # Add a small delay to ensure everything is rendered
                                    time.sleep(0.5)
                                    
                                    for i, rec in enumerate(valid_recommendations, 1):
                                        st.write(f"📸 Displaying outfit {i}...")
                                        display_outfit_recommendation(rec, existing_items, i)
                                        st.write(f"✅ Outfit {i} displayed successfully")
                                    
                                    st.success("🎊 All outfit recommendations have been generated and displayed!")
                                    st.balloons()
                                    
                                except Exception as e:
                                    st.error(f"Error generating recommendations: {e}")
                                    st.error("Raw response was:")
                                    # st.code(recommendation_str, language="text")
                        else:
                            st.warning("No relevant items found for your prompt. Try a different description.")
                else:
                    st.warning("Please enter a prompt for outfit recommendations.")
    
    # Tab 2: Chat Assistant
    with tab2:
        st.markdown('<div class="section-header">💬 Chat with Your Wardrobe Assistant</div>', unsafe_allow_html=True)
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Redisplay any stored images from previous interactions
            if 'displayed_images' in st.session_state and st.session_state.displayed_images:
                st.markdown("### 🖼️ Previously Found Items:")
                cols = st.columns(min(len(st.session_state.displayed_images), 3))
                
                for i, img_info in enumerate(st.session_state.displayed_images):
                    with cols[i % len(cols)]:
                        try:
                            # Display image with consistent frame dimensions
                            html_content = create_uniform_image_display(img_info['url'], img_info['name'], img_info['details'])
                            st.markdown(html_content, unsafe_allow_html=True)
                            
                        except Exception as e:
                            html_content = create_uniform_image_display("", img_info['name'], img_info['details'], "Image unavailable")
                            st.markdown(html_content, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input(
            "💭 Ask me about your wardrobe, fashion advice, or anything else:",
            placeholder="e.g., Show me all my formal shirts, What colors go well with navy blue?"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            send_button = st.button("Send", use_container_width=True)
        
        with col2:
            clear_chat = st.button("Clear Chat", use_container_width=True)
        
        if clear_chat:
            st.session_state.chat_history = []
            # Also clear displayed images
            if 'displayed_images' in st.session_state:
                st.session_state.displayed_images = []
            st.rerun()
        
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                try:
                    # Check if user is asking for specific items or general conversation
                    is_wardrobe_query = any(keyword in user_input.lower() for keyword in 
                                          ['show', 'find', 'search', 'wardrobe', 'clothes', 'outfit', 'shirt', 'pants', 'dress', 'shoe'])
                    
                    if is_wardrobe_query and existing_items:
                        # Use FAISS to find relevant items
                        if st.session_state.faiss_index is not None:
                            relevant_results = search_similar_items(
                                user_input, 
                                st.session_state.faiss_index, 
                                st.session_state.item_embeddings, 
                                embedding_model, 
                                existing_items,
                                top_k=10
                            )
                        else:
                            relevant_results = []
                        
                        if relevant_results:
                            # Create response with found items
                            response = f"I found {len(relevant_results)} relevant items in your wardrobe:\n\n"
                            
                            for result in relevant_results[:5]:  # Show top 5
                                item = result['details']
                                response += f"**{result['name']}**\n"
                                response += f"- Category: {item.get('category', 'N/A')}\n"
                                response += f"- Color: {item.get('color', 'N/A')}\n"
                                response += f"- Style: {item.get('style', 'N/A')}\n"
                                response += f"- Occasion: {item.get('occasion', 'N/A')}\n\n"
                            
                            # Add the response to chat history first
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            
                            # Display the response text
                            st.markdown(f"""
                            <div class="chat-message bot-message">
                                <strong>Assistant:</strong> {response}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show images in a grid - store this in session state
                            st.markdown("### 🖼️ Found Items:")
                            
                            # Store image display info in session state
                            if 'displayed_images' not in st.session_state:
                                st.session_state.displayed_images = []
                            
                            # Add current images to session state
                            current_images = []
                            cols = st.columns(min(len(relevant_results[:5]), 3))
                            
                            for i, result in enumerate(relevant_results[:5]):
                                with cols[i % len(cols)]:
                                    url = result['details'].get('url', '')
                                    if url:
                                        try:
                                            # Display image with consistent frame dimensions
                                            html_content = create_uniform_image_display(url, result['name'], result['details'])
                                            st.markdown(html_content, unsafe_allow_html=True)
                                            
                                            # Store image info for persistence
                                            current_images.append({
                                                'name': result['name'],
                                                'url': url,
                                                'details': result['details']
                                            })
                                            
                                        except Exception as e:
                                            st.write(f"Image unavailable for {result['name']}: {e}")
                                            html_content = create_uniform_image_display("", result['name'], result['details'], "Image unavailable")
                                            st.markdown(html_content, unsafe_allow_html=True)
                            
                            # Store the displayed images in session state
                            st.session_state.displayed_images = current_images
                            
                            # Debug info
                            st.write(f"📸 Stored {len(current_images)} images in session state")
                            st.write("Image URLs stored:")
                            for img in current_images:
                                st.write(f"- {img['name']}: {img['url'][:50]}...")
                            
                        else:
                            response = "I couldn't find any items matching your query. Try describing the items differently or add more items to your wardrobe."
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    else:
                        # General conversation using LLM
                        chat_prompt = ChatPromptTemplate.from_messages([
                            {"role": "system", "content": """You are a helpful fashion assistant. You can help with fashion advice, styling tips, color coordination, and general wardrobe questions. Be friendly, knowledgeable, and concise in your responses."""},
                            {"role": "user", "content": user_input}
                        ])
                        
                        chain = chat_prompt | llm | strout()
                        response = chain.invoke({"input": user_input})
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    error_response = f"I encountered an error: {e}. Please try again."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_response})
            
            st.rerun()
        
        # Show current recommendation memory summary
        with st.expander("🧠 Current Recommendation Memory", expanded=False):
            try:
                memory_vars = recommendation_memory.load_memory_variables({})
                conversation_summary = memory_vars.get("history", "")
                
                if conversation_summary:
                    st.write("**Previous Recommendations Summary:**")
                    st.info(conversation_summary)
                    
                    # Add clear memory button
                    if st.button("🗑️ Clear Recommendation Memory", type="secondary"):
                        recommendation_memory.clear()
                        st.success("Recommendation memory cleared!")
                        st.rerun()
                else:
                    st.write("No previous recommendations stored yet.")
            except Exception as e:
                st.write(f"No previous recommendations stored yet. (Error: {e})")
        
        # Show database items if available
        if existing_items:
            with st.expander("👗 Browse All Wardrobe Items", expanded=False):
                # Category filter
                categories = list(set(item.get('category', 'Unknown') for item in existing_items.values() if isinstance(item, dict)))
                selected_category = st.selectbox("Filter by category:", ["All"] + categories)
                
                # Filter items
                filtered_items = existing_items
                if selected_category != "All":
                    filtered_items = {name: item for name, item in existing_items.items() 
                                    if isinstance(item, dict) and item.get('category') == selected_category}
                
                # Display items in grid
                if filtered_items:
                    items_list = list(filtered_items.items())
                    cols = st.columns(3)
                    
                    for i, (name, details) in enumerate(items_list):
                        with cols[i % 3]:
                            url = details.get('url', '')
                            if url:
                                try:
                                    html_content = create_uniform_image_display(url, name, details)
                                    st.markdown(html_content, unsafe_allow_html=True)
                                    
                                except:
                                    html_content = create_uniform_image_display("", name, details, "Image unavailable")
                                    st.markdown(html_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()