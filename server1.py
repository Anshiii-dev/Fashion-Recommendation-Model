# app.py
import os
import json
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import re

from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# Config / constants
# -------------------------
load_dotenv(".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

BATCH_SIZE = 3
MAX_IMAGES_PER_BATCH = 5
MAX_RETRIES = 3

SUMMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
RECOMMENDATION_MODEL = "openai/gpt-oss-20b"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fashion-ai")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="Fashion Wardrobe AI API",
    description="Processes clothing images and generates outfit recommendations using ChromaDB and embeddings",
    version="1.0.0",
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -------------------------
# Pydantic models
# -------------------------
class AddItemsRequest(BaseModel):
    userid: str
    urls: List[HttpUrl]

class UserPreferences(BaseModel):
    eye_color: Optional[str] = "Not specified"
    body_type: Optional[str] = "Not specified"
    ethnicity: Optional[str] = "Not specified"
    temperature: Optional[str] = "Not specified"  # e.g., "25°C" or "77°F"

class RecommendationRequest(BaseModel):
    userid: str
    prompt: str
    num_recommendations: int = 3
    user_preferences: Optional[UserPreferences] = None

# -------------------------
# Application state
# -------------------------
class GroqKeyRotator:
    def __init__(self, api_keys: list, calls_per_key: int = 3):
        self.api_keys = api_keys
        self.calls_per_key = calls_per_key
        self.current_key_idx = 0
        self.call_count = 0

    def get_key(self):
        key = self.api_keys[self.current_key_idx]
        self.call_count += 1
        if self.call_count >= self.calls_per_key:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            self.call_count = 0
        return key

class AppState:
    def __init__(self):
        self.groq_key_rotator: Optional[GroqKeyRotator] = None
        self.llm: Optional[ChatGroq] = None
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None

    def initialize(self):
        # Collect all GROQ_API_KEY* variables from the environment
        api_keys = []
        for k, v in os.environ.items():
            if re.match(r"GROQ_API_KEY(\d+)?$", k) and v.strip():
                api_keys.append(v.strip())
        if not api_keys:
            raise ValueError("No GROQ_API_KEY* found in environment variables")
        self.groq_key_rotator = GroqKeyRotator(api_keys)

        # Initialize langchain groq chat LLM with the first key
        self.llm = ChatGroq(model=RECOMMENDATION_MODEL, temperature=0.8, api_key=self.groq_key_rotator.get_key())

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="wardrobe")

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("AppState initialized: GroqKeyRotator, ChromaDB, and SentenceTransformer ready")

state = AppState()

@app.on_event("startup")
async def on_startup():
    logger.info("Starting Fashion Wardrobe AI API...")
    state.initialize()
    logger.info("Startup complete")

# -------------------------
# Helper functions
# -------------------------
def _clean_json_response(text: str) -> str:
    """Remove ```json fences and surrounding backticks if present"""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def _call_vlm_for_batch(client: Groq, model: str, batch_urls: List[str]) -> Dict[str, Any]:
    """Call Groq VLM for a batch of image URLs"""
    urls_str = json.dumps(batch_urls)
    
    prompt = f"""═══════════════════════════════════════════════════════════════════
FASHION WARDROBE ANALYZER - IMAGE PROCESSING INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

ROLE: You are a professional fashion wardrobe analyzer and cataloger.

TASK: Analyze the provided clothing images and create a structured JSON catalog with descriptive metadata for each item.

═══════════════════════════════════════════════════════════════════
STEP-BY-STEP INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

1. EXAMINE EACH IMAGE
   → Carefully identify the type of clothing item(s) in each image
   → Look for multiple items if the image shows a complete outfit
   → Note all visible details: color, style, fabric, pattern, fit

2. CREATE DESCRIPTIVE NAMES
   → Format: "category_color_style_gender"
   → Examples: 
     • "top_white_oxford_shirt_male"
     • "bottom_black_denim_jeans_female"
     • "footwear_brown_leather_boots_male"
   → Make names unique and descriptive
   → Include gender specification to avoid confusion

3. CATEGORIZE ITEMS
   → Categories: top, bottom, footwear, accessory, outerwear
   → Be specific: Choose ONE primary category per item
   → If multiple items visible, create separate entries

4. EXTRACT DETAILED METADATA
   → Fill ALL attributes for each item
   → Be specific and descriptive (avoid generic terms)
   → Ensure gender is clearly specified
5. Readable Image names:
Generate a 2nd name for image use this name and variable name for it
the variable name is "readable_image_name"
for 2nd name  use this format for image names:
 Format:
   [Color] [Gender] [Category/Genre] [Material or Style]
Guidelines:
 → Keep names short, natural, and descriptive (2–5 words).
 → Use real-world fashion terms (e.g., “crewneck,” “slim-fit,” “linen,” “oxford”).
 → Always include gender: “men’s” or “women’s” (or “unisex” if neutral).
 → Avoid redundancy (don’t repeat “shirt” twice, etc.).
 → Make each name sound aesthetically appealing and ready for e-commerce display.
Examples:
 “White men’s oxford shirt”
 “Black women’s denim jeans” 
 “Brown unisex leather boots”
 “Beige women’s linen trench coat”
 “Navy men’s slim-fit chinos”
═══════════════════════════════════════════════════════════════════
IMAGE URLS TO ANALYZE
═══════════════════════════════════════════════════════════════════

{urls_str}

CRITICAL: Use the EXACT URLs above - DO NOT modify or create example URLs!

═══════════════════════════════════════════════════════════════════
REQUIRED JSON RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

{{
  "descriptive_item_name": {{
    "readable_name": "A human-readable name, e.g., 'Navy Men's Oxford Shirt'",
    "url": "EXACT_URL_FROM_LIST_ABOVE",
    "category": "top/bottom/footwear/accessory/outerwear",
    "color": "specific color name (e.g., navy blue, burgundy, charcoal gray)",
    "genre": "specific type (e.g., oxford shirt, denim jeans, leather boots)",
    "style": "style description (e.g., casual button-down, slim-fit, ankle-length)",
    "gender": "male/female",
    "season": "spring/summer/fall/winter",
    "fashion_type": "casual/formal/business casual/sporty/elegant/streetwear",
    "design": "design elements (e.g., minimalist, embroidered, distressed)",
    "fabric": "fabric type (e.g., cotton, denim, leather, polyester blend)",
    "pattern": "pattern type (e.g., solid, striped, plaid, floral, polka dot)",
    "fit": "fit description (e.g., slim fit, relaxed fit, tailored, oversized)",
    "occasion": "suitable occasions (e.g., office, party, casual outing, gym)"
  }}
}}

═══════════════════════════════════════════════════════════════════
EXAMPLE (for reference only)
═══════════════════════════════════════════════════════════════════

{{
  "top_navy_oxford_shirt_male": {{
    "readable_name": "Navy Men's Oxford Shirt",
    "url": "https://example.com/image1.jpg",
    "category": "top",
    "color": "navy blue",
    "genre": "oxford shirt",
    "style": "classic button-down collar",
    "gender": "male",
    "season": "spring",
    "fashion_type": "business casual",
    "design": "minimalist with chest pocket",
    "fabric": "100% cotton oxford cloth",
    "pattern": "solid",
    "fit": "slim fit",
    "occasion": "office, smart casual events"
  }}
}}

═══════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS - MUST FOLLOW
═══════════════════════════════════════════════════════════════════

OUTPUT FORMAT:
   • Response must be ONLY valid JSON
   • NO markdown formatting (no ```json or ```)
   • NO explanations or extra text
   • Start with {{ and end with }}
   • Properly escape special characters

DATA ACCURACY:
   • Use EXACT URLs from the list above
   • Match URLs to images in the order provided
   • Create unique, descriptive names for each item
   • Be specific - avoid generic descriptions like "nice" or "good"

GENDER CLARITY:
   • Always specify male/female clearly
   • Ensure downstream models won't be confused
   • Keep male and female items distinctly labeled

COMPLETE OUTFITS:
   • If image shows multiple items (complete outfit):
     → Extract EACH item separately
     → Create individual entries for top, bottom, footwear, accessories
     → Link them conceptually but list separately
   
METADATA COMPLETENESS:
   • Fill ALL fields for each item
   • Use descriptive, specific terms
   • Ensure season matches the item's appropriate use

═══════════════════════════════════════════════════════════════════
BEGIN ANALYSIS
═══════════════════════════════════════════════════════════════════

Analyze the images and return ONLY the JSON object. Start now:
"""

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

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        timeout=60
    )

    raw = resp.choices[0].message.content.strip()
    cleaned = _clean_json_response(raw)
    parsed = json.loads(cleaned)
    return parsed

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
        logger.error(f"Error loading existing items: {e}")
        return {}

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Fashion Wardrobe AI",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/items/add")
async def add_items(request: AddItemsRequest, background_tasks: BackgroundTasks):
    """
    Process clothing images using VLM and add them to ChromaDB.
    Returns parsed items as JSON.
    """
    client = get_rotated_groq_client()
    collection = state.collection
    
    if not client or not collection:
        raise HTTPException(status_code=500, detail="Server not properly initialized")

    urls = [str(u) for u in request.urls]
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    effective_batch_size = min(BATCH_SIZE, MAX_IMAGES_PER_BATCH)
    all_items: Dict[str, Any] = {}
    total = len(urls)
    total_batches = (total + effective_batch_size - 1) // effective_batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * effective_batch_size
        end = min(start + effective_batch_size, total)
        batch_urls = urls[start:end]

        attempt = 0
        success = False
        last_exception = None

        while attempt < MAX_RETRIES and not success:
            try:
                logger.info(f"Processing batch {batch_idx+1}/{total_batches}, attempt {attempt+1}")
                batch_items = _call_vlm_for_batch(client, SUMMARY_MODEL, batch_urls)

                # Add to ChromaDB
                for item_name, item_details in batch_items.items():
                    # Create document for embedding
                    document = f"{item_details.get('category', '')} {item_details.get('color', '')} {item_details.get('style', '')} {item_details.get('gender', '')} {item_details.get('season', '')} {item_details.get('fashion_type', '')} {item_details.get('design', '')} {item_details.get('fabric', '')} {item_details.get('pattern', '')} {item_details.get('fit', '')} {item_details.get('occasion', '')}"
                    
                    # Add userid to metadata for data isolation
                    item_details['userid'] = request.userid
                    
                    # Add to collection
                    collection.add(
                        ids=[item_name],
                        documents=[document],
                        metadatas=[item_details]
                    )
                    
                    all_items[item_name] = item_details

                success = True
                logger.info(f"Batch {batch_idx+1} processed successfully ({len(batch_items)} items)")
            except Exception as e:
                last_exception = e
                attempt += 1
                logger.error(f"Error processing batch {batch_idx+1} attempt {attempt}: {e}")
                if attempt >= MAX_RETRIES:
                    logger.error(f"Giving up on batch {batch_idx+1} after {MAX_RETRIES} attempts")
                    raise HTTPException(status_code=500, detail=f"Failed to process batch {batch_idx+1}: {last_exception}")

    return {
        "success": True,
        "items_added": len(all_items),
        "items": all_items,
        "message": f"Successfully processed and added {len(all_items)} items to database"
    }

@app.get("/items/all/{userid}")
async def get_all_items(userid: str):
    """
    Retrieve all wardrobe items from ChromaDB for a specific user.
    """
    collection = state.collection
    
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")
    
    try:
        # Query all items for this user only
        all_items_result = collection.get(where={"userid": userid})
        existing_items = {}
        
        if all_items_result["ids"]:
            for i, item_id in enumerate(all_items_result["ids"]):
                metadata = all_items_result["metadatas"][i]
                existing_items[item_id] = metadata
        
        # Get category statistics
        categories = {}
        for item in existing_items.values():
            if isinstance(item, dict):
                cat = item.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "success": True,
            "total_items": len(existing_items),
            "categories": categories,
            "items": existing_items,
            "message": f"Retrieved {len(existing_items)} items from database for user {userid}"
        }
    except Exception as e:
        logger.error(f"Error retrieving items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve items: {e}")


@app.get("/items/all")
async def get_all_items_global():
    """
    Retrieve all wardrobe items from ChromaDB across all users.
    """
    collection = state.collection
    
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")
    
    try:
        # Query all items (no userid filter)
        all_items_result = collection.get()
        existing_items = {}
        
        if all_items_result["ids"]:
            for i, item_id in enumerate(all_items_result["ids"]):
                metadata = all_items_result["metadatas"][i]
                existing_items[item_id] = metadata
        
        # Get category statistics
        categories = {}
        for item in existing_items.values():
            if isinstance(item, dict):
                cat = item.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "success": True,
            "total_items": len(existing_items),
            "categories": categories,
            "items": existing_items,
            "message": f"Retrieved {len(existing_items)} items from database (all users)"
        }
    except Exception as e:
        logger.error(f"Error retrieving all items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve items: {e}")

@app.post("/recommendations/get")
async def get_recommendations(request: RecommendationRequest):
    """
    Generate outfit recommendations using ChromaDB query and embeddings.
    Uses the same approach as the Streamlit app.
    """
    if not state.collection:
        raise HTTPException(status_code=500, detail="Server not properly initialized")

    collection = state.collection
    llm = get_rotated_llm()

    # Query ChromaDB for relevant items for this user
    try:
        query_results = collection.query(
            query_texts=[request.prompt],
            n_results=30,
            where={"userid": request.userid}
        )
        
        relevant_items = {}
        if query_results["ids"][0]:
            for i in range(len(query_results["ids"][0])):
                item_name = query_results["ids"][0][i]
                metadata = query_results["metadatas"][0][i]
                relevant_items[item_name] = metadata
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query database: {e}")

    if not relevant_items:
        raise HTTPException(status_code=404, detail="No relevant items found for your prompt")

    prefs = request.user_preferences or UserPreferences()

    # Build recommendation prompt with clear formatting
    recommendation_prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": """═══════════════════════════════════════════════════════════════════
PROFESSIONAL FASHION STYLIST - OUTFIT RECOMMENDATION SYSTEM
═══════════════════════════════════════════════════════════════════

ROLE: You are an expert personal fashion stylist creating personalized outfit recommendations.

REQUEST: Create {num_recommendations} complete outfit recommendations for: "{prompt}"

═══════════════════════════════════════════════════════════════════
AVAILABLE WARDROBE ITEMS
═══════════════════════════════════════════════════════════════════
{context}
═══════════════════════════════════════════════════════════════════
WEATHER CONDITIONS (HIGHEST PRIORITY)
═══════════════════════════════════════════════════════════════════
Current Temperature: {temperature}
YOU MUST STRICTLY FOLLOW THESE TEMPERATURE-BASED GUIDELINES:
┌─────────────────────────────────────────────────────────────────┐
│ HOT WEATHER: Above 25°C (77°F)                                  │
├─────────────────────────────────────────────────────────────────┤
│ RECOMMEND:                                                      │
│    • Lightweight, breathable tops (cotton, linen, rayon)        │
│    • Shorts, skirts, summer dresses                             │
│    • Sandals, canvas shoes, breathable footwear                 │
│    • Light, airy fabrics                                        │
│                                                                  │
│ STRICTLY AVOID:                                                  │
│    • Jackets, blazers, cardigans                                │
│    • Sweaters, hoodies                                          │
│    • Heavy outerwear of any kind                                │
│    • Thick fabrics, wool, denim jackets                         │
│    • Boots, closed heavy shoes                                  │
│                                                                  │
│ Season Filter: ONLY "summer" items                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ MILD WEATHER: 15°C to 25°C (59°F to 77°F)                       │
├─────────────────────────────────────────────────────────────────┤
│ RECOMMEND:                                                      │
│    • Light layering: T-shirts, long sleeves                     │
│    • Jeans, chinos, casual pants                                │
│    • Sneakers, loafers, casual shoes                            │
│    • Optional light cardigan/jacket (only near 15°C)            │
│                                                                  │
│ NOTES:                                                          │
│    • Outerwear is OPTIONAL (add only if temp ≤ 18°C)            │
│    • If added, must be lightweight (windbreaker, light jacket)  │
│                                                                  │
│ Season Filter: "spring" or "fall" items preferred               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COOL WEATHER: 5°C to 15°C (41°F to 59°F)                        │
├─────────────────────────────────────────────────────────────────┤
│ RECOMMEND:                                                      │
│    • Long sleeve shirts, sweaters, pullovers                    │
│    • Long pants, jeans                                          │
│    • Closed-toe shoes, boots                                    │
│    • Light to medium jacket or coat                             │
│                                                                  │
│ REQUIRED:                                                       │
│    • MUST include light to medium outerwear                     │
│                                                                  │
│ Season Filter: "fall" or "winter" items                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COLD WEATHER: Below 5°C (41°F)                                  │
├─────────────────────────────────────────────────────────────────┤
│ RECOMMEND:                                                      │
│    • Thick sweaters, turtlenecks, thermal layers                │
│    • Long pants, thick jeans, wool trousers                     │
│    • Boots, warm closed shoes                                   │
│    • Heavy coat, winter jacket, parka                           │
│    • Warm accessories (scarves, gloves recommended)             │
│                                                                  │
│ REQUIRED:                                                       │
│    • MUST include heavy winter outerwear                        │
│                                                                  │
│ STRICTLY AVOID:                                                 │
│    • Shorts, skirts, sleeveless items                           │
│    • Sandals, open-toe shoes                                    │
│    • Light, thin fabrics                                        │
│                                                                  │
│ Season Filter: ONLY "winter" items                              │
└─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
USER PERSONAL PREFERENCES (Secondary Priority)
═══════════════════════════════════════════════════════════════════

• Eye Color:   {eye_color}
• Body Type:   {body_type}
• Ethnicity:   {ethnicity}

NOTE: Use these to personalize recommendations, but they should complement 
      (not override) the temperature requirements.

═══════════════════════════════════════════════════════════════════
OUTFIT COMPOSITION REQUIREMENTS
═══════════════════════════════════════════════════════════════════

Each complete outfit MUST contain:

REQUIRED ITEMS:
   • Top (1 item) - shirt, t-shirt, blouse, dress top
   • Bottom (1 item) - pants, jeans, skirt, shorts (if appropriate for temp)
   • Footwear (1 item) - shoes, boots, sandals (matching weather)

CONDITIONAL ITEMS:
   • Outerwear:
     - REQUIRED if temperature < 15°C
     - FORBIDDEN if temperature > 25°C
     - OPTIONAL if temperature 15-25°C (only near 15°C)

OPTIONAL ITEMS:
   • Accessories (watch, bag, jewelry, scarf)
   • Only include if they enhance the outfit
   • Must complement the overall style

═══════════════════════════════════════════════════════════════════
CRITICAL GENDER RULES
═══════════════════════════════════════════════════════════════════

• Be 100% specific about gender


ANTI-REPETITION RULES (Analyze history carefully):

1. NEVER repeat the exact same outfit combinations
2. NEVER reuse the same specific clothing items from recent recommendations
3. AVOID similar color schemes already suggested
4. If prompt is similar to previous, choose DIFFERENT items
5. Create diverse combinations from available wardrobe
6. Only when ALL combinations are exhausted, then you may mix/repeat

═══════════════════════════════════════════════════════════════════
RECOMMENDATION PRIORITY ORDER
═══════════════════════════════════════════════════════════════════

1. FIRST PRIORITY:  Temperature Guidelines (mandatory compliance)
2. SECOND PRIORITY: User's Prompt Requirements (formal/casual/occasion)
3. THIRD PRIORITY:  Personal Preferences (eye color, body type, ethnicity)

═══════════════════════════════════════════════════════════════════
OUTFIT COMPOSITION REQUIREMENTS
═══════════════════════════════════════════════════════════════════

Each complete outfit MUST contain:

REQUIRED ITEMS:
   • Top (1 item) - shirt, t-shirt, blouse, dress top
   • Bottom (1 item) - pants, jeans, skirt, shorts (if appropriate for temp)
   • Footwear (1 item) - shoes, boots, sandals (matching weather)

CONDITIONAL ITEMS:
   • Outerwear:
     - REQUIRED if temperature < 15°C
     - FORBIDDEN if temperature > 25°C
     - OPTIONAL if temperature 15-25°C (only near 15°C)

OPTIONAL ITEMS:
   • Accessories (watch, bag, jewelry, scarf)
   • Only include if they enhance the outfit
   • Must complement the overall style

═══════════════════════════════════════════════════════════════════
CRITICAL GENDER RULES
═══════════════════════════════════════════════════════════════════

• Be 100% specific about gender
• Be 100% specific about gender in ALL recommendations
• NO mixing of male and female items in the same outfit
• Female items → Female outfits ONLY
• Male items → Male outfits ONLY
• Clearly state gender in the recommendation description

═══════════════════════════════════════════════════════════════════
CRITICAL - REQUIRED JSON OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

Your response MUST be a valid JSON array. Each object in the array represents one outfit recommendation.

Each outfit object MUST contain the following three fields:
1.  `recommendation`: A string describing the full outfit.
2.  `reason`: A string explaining why the outfit is suitable.
3.  `image_names`: An array of strings.
4.  `readable_image_names`: An array of strings.
5.  `missing_items`: An array of strings for items needed to complete the outfit that are not in the wardrobe.

**RULES FOR `image_names` vs. `readable_image_names`:**

1.  **`image_names` (FOR THE SYSTEM):**
    *   **MUST** contain the **EXACT, UNMODIFIED** item names (the keys) from the "AVAILABLE WARDROBE ITEMS" list.
    *   This is for internal system use. **DO NOT** alter these keys.
    *   Example: `["top_black_button_down_shirt_male", "bottom_green_cargo_pants_male"]`

2.  **`readable_image_names` (FOR DISPLAY):**
    *   **MUST** contain beautiful, human-readable names for each corresponding item in `image_names`.
    *   This is for display to the user.
    *   **Format:** `[Color] [Gender] [Category/Genre] [Material or Style]`
    *   **Guidelines:**
        *   Keep names short, natural, and descriptive (2–5 words).
        *   Use real-world fashion terms (e.g., “crewneck,” “slim-fit,” “linen,” “oxford”).
        *   Always include gender: “men’s” or “women’s”.
        *   Make each name sound appealing for e-commerce.
    *   Example: `["Black Men's Button-Down Shirt", "Green Men's Cargo Pants"]`

**The two arrays MUST have the same number of items and correspond to each other.**
The response should be valid JSON without any markdown formatting or extra text.
[
  {{
    "recommendation": "A stylish and comfortable outfit for a mild day...",
    "reason": "This outfit is perfect for the current temperature because...",
    "image_names": ["top_black_button_down_shirt_male", "bottom_green_cargo_pants_male", "footwear_white_sneakers_male"],
    "readable_image_names": ["Black Men's Button-Down Shirt", "Green Men's Cargo Pants", "White Men's Sneakers"],
    "missing_items": ["sunglasses"]
  }}
]

═══════════════════════════════════════════════════════════════════
FINAL PRE-SUBMISSION CHECKLIST
═══════════════════════════════════════════════════════════════════

Before submitting your response, verify:

□ Response starts with [ and ends with ]
□ Valid JSON format with proper escaping
□ Each item's season matches the temperature:
  ↳ Hot (>25°C): Only summer items
  ↳ Cold (<5°C): Only winter items
  ↳ Mild: Spring/fall items
□ Outerwear rules followed:
  ↳ Included if temp < 15°C
  ↳ Excluded if temp > 25°C
□ All items exist in available wardrobe
□ No gender mixing in outfits
□ Color coordination considered
□ Style matching verified
□ NO markdown formatting (no ```json or ```)
□ NO explanatory text outside JSON
□ Temperature mentioned in recommendations

═══════════════════════════════════════════════════════════════════
BEGIN CREATING RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════

Generate ONLY the JSON array now. No other text:
"""}
    ])

    # Create chain
    chain = recommendation_prompt | llm | StrOutputParser()    # Invoke chain
    try:
        recommendation_str = chain.invoke({
            "prompt": request.prompt+"It should include a top,a bottom and footwear.you can also add acessory if it looks good.",
            "context": json.dumps(relevant_items, indent=2),
            "num_recommendations": request.num_recommendations,
            # "history": history,
            "temperature": prefs.temperature,
            "eye_color": prefs.eye_color,
            "body_type": prefs.body_type,
            "ethnicity": prefs.ethnicity
        })
    except Exception as e:
        logger.error(f"LLM chain invocation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM invocation error: {e}")

    cleaned = _clean_json_response(recommendation_str)

    try:
        recommendations = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM recommendation JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response JSON: {e}")

    if not isinstance(recommendations, list):
        raise HTTPException(status_code=500, detail="LLM response was not a JSON array")

    logger.info(f"DEBUG: Relevant item keys available for URL resolution: {list(relevant_items.keys())}")

    # Resolve outfit_urls from existing_items metadata
    for rec in recommendations:
        resolved_urls = []
        image_names = rec.get("image_names", []) or []
        logger.info(f"DEBUG: LLM returned image_names: {image_names}")
        for name in image_names:
            item_meta = relevant_items.get(name)
            if isinstance(item_meta, dict):
                url = item_meta.get("url")
                if url:
                    resolved_urls.append(url)
                    logger.info(f"DEBUG: Found URL for '{name}'")
                else:
                    logger.warning(f"DEBUG: Found item '{name}' but it has no URL.")
            else:
                logger.warning(f"DEBUG: Could not find item '{name}' in relevant_items.")
        rec["outfit_urls"] = resolved_urls

    return {
        "success": True,
        "recommendations": recommendations,
        "total_items_in_wardrobe": collection.count(),
        "relevant_items_found": len(relevant_items),
        "message": f"Generated {len(recommendations)} outfit recommendations"
    }

@app.delete("/database/clear")
async def clear_database():
    """
    DANGEROUS: Deletes the entire 'wardrobe' collection and recreates it.
    This will wipe ALL data for ALL users. Use with extreme caution.
    """
    collection_name = "wardrobe"
    try:
        # Get the count of items before deleting
        num_items_deleted = state.collection.count()
        logger.warning(f"Received request to clear entire database. Deleting collection: {collection_name} which has {num_items_deleted} items.")
        
        state.chroma_client.delete_collection(name=collection_name)
        state.collection = state.chroma_client.get_or_create_collection(name=collection_name)
        logger.info(f"Successfully deleted and recreated collection: {collection_name}")
        
        return {
            "success": True,
            "items_deleted": num_items_deleted,
            "message": f"Successfully deleted {num_items_deleted} items and recreated collection '{collection_name}'. All data has been wiped."
        }
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@app.delete("/items/delete-by-names/{userid}")
async def delete_items_by_names(userid: str, image_names: list = Body(..., embed=True)):
    """
    Delete items from the wardrobe database by their image_names for a specific user.
    Only deletes items that belong to the specified user.
    """
    collection = state.collection
    if not collection:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")
    deleted_count = 0
    for name in image_names:
        try:
            # Verify the item belongs to this user before deleting
            item = collection.get(ids=[name], where={"userid": userid})
            if item["ids"]:
                collection.delete(ids=[name])
                deleted_count += 1
                logger.info(f"Deleted item '{name}' for user {userid}")
            else:
                logger.warning(f"Item '{name}' not found for user {userid}")
        except Exception as e:
            logger.error(f"Error deleting item '{name}' for user {userid}: {e}")
    return {
        "success": True,
        "items_deleted": deleted_count,
        "message": f"Deleted {deleted_count} items from database for user {userid}."
    }

# --- Helper to get a fresh Groq client with rotated key ---
def get_rotated_groq_client():
    key = state.groq_key_rotator.get_key()
    logger.info(f"[GROQ KEY ROTATION] Using Groq API key index {state.groq_key_rotator.current_key_idx} ({key[:6]}...)")
    return Groq(api_key=key)

def get_rotated_llm():
    key = state.groq_key_rotator.get_key()
    logger.info(f"[GROQ KEY ROTATION] Using Groq API key index {state.groq_key_rotator.current_key_idx} ({key[:6]}...)")
    return ChatGroq(model=RECOMMENDATION_MODEL, temperature=0.8, api_key=key)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)