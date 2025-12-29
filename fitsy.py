import streamlit as st
import json
import os
import time
import requests
import uuid
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from dotenv import load_dotenv
import re
from PIL import Image
import io

# API Configuration
API_URL = "http://localhost:8000"

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
    
    # Load sentence transformer for FAISS
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return embedding_model

# FAISS operations
def create_faiss_index(items, embedding_model):
    """Create FAISS index from wardrobe items"""
    import faiss
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
    import faiss
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
    display_name = details.get('readable_name', name) if isinstance(details, dict) else name
    try:
        if url:
            # Ensure consistent image dimensions
            processed_url = ensure_image_dimensions(url)
            return f"""
            <div class="uniform-image-container">
                <img src="{processed_url}" alt="{display_name}" style="max-width: 100%; max-height: 300px; object-fit: contain;">
                <div class="item-details">
                    <strong>{display_name}</strong><br>
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
                    <strong>{display_name}</strong><br>
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
                <strong>{display_name}</strong><br>
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

def load_existing_items():
    """Load existing items from Backend API"""
    try:
        response = requests.get(f"{API_URL}/items/all")
        if response.status_code == 200:
            data = response.json()
            return data.get("items", {})
        else:
            st.error(f"Failed to load items from server: {response.status_code}")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {e}")
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
    readable_names = rec.get('readable_image_names', [])
    
    if image_names:
        st.subheader("👕 Recommended Items")
        
        # Create columns for images
        cols = st.columns(min(len(image_names), 4))
        
        for i, name in enumerate(image_names):
            with cols[i % len(cols)]:
                if name in items:
                    item_details = items[name]
                    url = item_details.get('url', '')
                    
                    # Determine display name
                    display_name = name
                    if readable_names and i < len(readable_names):
                        display_name = readable_names[i]
                    
                    # Update details with readable name for display
                    temp_details = item_details.copy() if isinstance(item_details, dict) else {}
                    temp_details['readable_name'] = display_name
                    
                    if url:
                        # Use the cached image loading function
                        load_and_display_image(url, display_name, display_name, temp_details)
                else:
                    st.warning(f"Item '{name}' not found in database")
    
    # Add Try-On Button
    st.divider()
    
    # Create a unique container for this outfit's try-on section
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(f"👤 Try On Outfit {index}", key=f"tryon_btn_{index}", use_container_width=True):
                st.session_state[f"show_tryon_input_{index}"] = not st.session_state.get(f"show_tryon_input_{index}", False)
        
        # Show try-on input form if button was clicked
        if st.session_state.get(f"show_tryon_input_{index}", False):
            st.markdown("### 👤 Virtual Try-On")
            st.info("Provide a photo of yourself to see how this outfit looks on you!")
            
            # Two options: Upload from local storage OR use URL
            input_method = st.radio("Choose how to provide your image:", ["📤 Upload from Device", "🔗 Use Image URL"], key=f"input_method_{index}", horizontal=False)
            
            person_image_url = None
            
            if input_method == "📤 Upload from Device":
                # File uploader for local image storage
                uploaded_image = st.file_uploader(
                    "Select your photo from your device:",
                    type=["jpg", "jpeg", "png", "gif", "webp"],
                    key=f"person_image_upload_{index}"
                )
                
                if uploaded_image is not None:
                    # Show preview of uploaded image
                    st.image(uploaded_image, caption="Your photo preview", use_column_width=True)
                    
                    # Save uploaded image and create URL
                    try:
                        timestamp = int(time.time() * 1000)
                        safe_filename = f"{timestamp}_person_{uploaded_image.name}"
                        file_path = os.path.join("static/uploads", safe_filename)
                        
                        # Ensure directory exists
                        os.makedirs("static/uploads", exist_ok=True)
                        
                        # Save the uploaded file
                        with open(file_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())
                        
                        # Create local URL
                        person_image_url = f"http://localhost:8000/static/uploads/{safe_filename}"
                        st.success(f"✅ Image uploaded and saved!")
                        
                    except Exception as e:
                        st.error(f"Error saving uploaded image: {e}")
            
            else:  # Use Image URL
                person_image_input = st.text_input(
                    "Enter person image URL:",
                    placeholder="e.g., http://localhost:8000/static/uploads/person.jpg",
                    key=f"person_url_{index}"
                )
                if person_image_input:
                    person_image_url = person_image_input
                    # Show preview if URL is provided
                    try:
                        st.image(person_image_url, caption="Image preview", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not preview image: {e}")
            
            col_generate, col_cancel = st.columns(2)
            
            with col_generate:
                if st.button("🎨 Generate Try-On Images", key=f"generate_tryon_{index}", use_container_width=True):
                    if person_image_url:
                        with st.spinner("🔄 Generating virtual try-on images... This may take a moment..."):
                            try:
                                # Read the image file and send as multipart/form-data
                                image_file_path = None
                                
                                # If it's an uploaded file, find its path
                                if use_upload and uploaded_image:
                                    safe_filename = f"temp_{uuid.uuid4().hex}.png"
                                    file_path = f"static/uploads/{safe_filename}"
                                    image_file_path = file_path
                                elif person_image_url.startswith("http"):
                                    # For URLs, we need to download and save locally first
                                    response_img = requests.get(person_image_url)
                                    if response_img.status_code == 200:
                                        safe_filename = f"temp_{uuid.uuid4().hex}.png"
                                        file_path = f"static/uploads/{safe_filename}"
                                        with open(file_path, "wb") as f:
                                            f.write(response_img.content)
                                        image_file_path = file_path
                                
                                if image_file_path:
                                    # Use multipart form-data to avoid URL length limits
                                    with open(image_file_path, "rb") as img_file:
                                        files = {"person_image": img_file}
                                        data = {"recommendation_image_names": image_names}
                                        response = requests.post(
                                            f"{API_URL}/tryon/generate",
                                            files=files,
                                            data=data
                                        )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        
                                        if result.get("success"):
                                            st.success("✅ Virtual try-on images generated successfully!")
                                            
                                            # Display the try-on images
                                            st.markdown("### 🎯 Try-On Results")
                                            
                                            col_left, col_right = st.columns(2)
                                        
                                        with col_left:
                                            st.markdown("**Left Side View**")
                                            st.image(result.get("left_side_view_url"), use_column_width=True)
                                        
                                        with col_right:
                                            st.markdown("**Right Side View**")
                                            st.image(result.get("right_side_view_url"), use_column_width=True)
                                        
                                        # Show outfit details
                                        st.markdown("### 📋 Outfit Details")
                                        for i, item in enumerate(result.get("outfit_items", []), 1):
                                            st.write(f"{i}. {item}")
                                        
                                        st.balloons()
                                        
                                        # Keep the form open to show results
                                    else:
                                        st.error(f"Failed to generate try-on: {result.get('detail', 'Unknown error')}")
                                else:
                                    st.error(f"Server error: {response.status_code} - {response.text}")
                            
                            except Exception as e:
                                st.error(f"Error generating try-on images: {e}")
                    else:
                        st.warning("Please upload an image or provide a valid image URL")
            
            with col_cancel:
                if st.button("❌ Close Try-On", key=f"cancel_tryon_{index}", use_container_width=True):
                    st.session_state[f"show_tryon_input_{index}"] = False

# Configuration
BATCH_SIZE = 3  # Process 3 images at a time (within the 5 image limit)
MAX_RETRIES = 3  # Maximum retry attempts for failed batches

# Ensure batch size doesn't exceed API limit
if BATCH_SIZE > 5:
    st.warning("⚠️ Batch size reduced to 5 (API limit)")
    BATCH_SIZE = 5

def process_uploaded_files(uploaded_files):
    """Process uploaded files via Backend API"""
    if not uploaded_files:
        return {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Uploading and processing images...")
    
    try:
        files = []
        for file in uploaded_files:
            files.append(('files', (file.name, file.getvalue(), file.type)))
        
        response = requests.post(f"{API_URL}/items/add", files=files)
        
        if response.status_code == 200:
            result = response.json()
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            return result.get("items", {})
        else:
            st.error(f"Server error: {response.text}")
            return {}
            
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return {}

def main():
    # Initialize
    init_session_state()
    
    # Load models and initialize database
    embedding_model = load_env_and_models()
    # chroma_client, collection = init_chromadb() # Removed as we use API
    
    # Header
    st.markdown('<h1 class="main-header">👗 Fashion Wardrobe AI Assistant</h1>', unsafe_allow_html=True)
    
    # Load existing items
    existing_items = load_existing_items()
    
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
        
        uploaded_files = st.file_uploader(
            "Upload clothing images:",
            type=['png', 'jpg', 'jpeg', 'webp'],
            accept_multiple_files=True
        )
        
        if st.button("🚀 Process New Items", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing images with Gemini..."):
                    new_items = process_uploaded_files(uploaded_files)
                    
                if new_items:
                    st.success(f"✅ Added {len(new_items)} new items!")
                    existing_items.update(new_items)
                    
                    # Update FAISS index
                    st.session_state.faiss_index, st.session_state.item_embeddings = create_faiss_index(existing_items, embedding_model)
            else:
                st.error("❌ No files uploaded!")

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
                
                temperature = st.text_input(
                    "🌡️ Current Temperature (optional):",
                    placeholder="e.g., 25°C, 75°F, Cold, Hot"
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
                        try:
                            # Prepare payload for Backend API
                            payload = {
                                "prompt": prompt,
                                "num_recommendations": num_recommendations,
                                "user_preferences": {
                                    "eye_color": st.session_state.eye_color,
                                    "body_type": st.session_state.body_type,
                                    "ethnicity": st.session_state.ethnicity,
                                    "temperature": temperature if temperature else "Not specified"
                                }
                            }
                            
                            # Call Backend API
                            response = requests.post(f"{API_URL}/recommendations/get", json=payload)
                            
                            if response.status_code == 200:
                                result = response.json()
                                recommendations = result.get("recommendations", [])
                                
                                if recommendations:
                                    st.write(f"🎉 Generated {len(recommendations)} outfit recommendations!")
                                    
                                    # Add a small delay to ensure everything is rendered
                                    time.sleep(0.5)
                                    
                                    for i, rec in enumerate(recommendations, 1):
                                        st.write(f"📸 Displaying outfit {i}...")
                                        display_outfit_recommendation(rec, existing_items, i)
                                        st.write(f"✅ Outfit {i} displayed successfully")
                                    
                                    st.success("🎊 All outfit recommendations have been generated and displayed!")
                                    st.balloons()
                                else:
                                    st.warning("No recommendations generated. Try a different prompt.")
                            else:
                                st.error(f"Server error: {response.status_code} - {response.text}")
                                
                        except Exception as e:
                            st.error(f"Error connecting to server: {e}")
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
            "💭 Search your wardrobe:",
            placeholder="e.g., Show me all my formal shirts, Find my blue jeans..."
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
        
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Searching..."):
                try:
                    if existing_items:
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
                            
                        else:
                            response = "I couldn't find any items matching your query. Try describing the items differently or add more items to your wardrobe."
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        st.warning("No items in wardrobe to search.")
                
                except Exception as e:
                    error_response = f"I encountered an error: {e}. Please try again."
                    st.session_state.chat_history.append({"role": "assistant", "content": error_response})
        
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