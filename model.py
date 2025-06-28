from flask import Flask, request, render_template, jsonify
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import numpy as np
from sklearn.cluster import KMeans
from pytrends.request import TrendReq
import requests

app = Flask(__name__)

# ----------------------------
# Setup: CLIP Model
# ----------------------------
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------------
# Config: Google CSE
# ----------------------------
GOOGLE_API_KEY = "AIzaSyACSK1-Liz4zCfc_4nRf4Z-fw0VeKw2tj4"
GOOGLE_CSE_ID = "a736a57b0897c4c58"

# ----------------------------
# Trend Text Generation (via pytrends)
# ----------------------------
def fetch_trending_fashion_terms(keywords, geo='IN', top_n=4):
    pytrends = TrendReq()
    pytrends.build_payload(keywords, timeframe='now 7-d', geo=geo)
    trends = pytrends.interest_over_time()
    if trends.empty:
        return keywords[:top_n]
    return trends.mean().sort_values(ascending=False).head(top_n).index.tolist()

# ----------------------------
# Fetch Trend Image via Google CSE
# ----------------------------
import requests
from PIL import Image
import io
import requests
from PIL import Image
import io

def clean_query(text):
    """
    Remove extra spaces, newlines, and hidden characters from the trend query.
    """
    return " ".join(text.strip().split())

def fetch_trend_image(trend_query):
    try:
        # Step 1: Clean and log the query
        trend_query = clean_query(trend_query)
        print(f"\n📡 Querying Google Image for: '{trend_query}'")

        # Step 2: Set credentials
        api_key = "AIzaSyACSK1-Liz4zCfc_4nRf4Z-fw0VeKw2tj4"
        cx = "a736a57b0897c4c58"

        # Step 3: Build the API URL manually
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?q={trend_query}&cx={cx}&key={api_key}&searchType=image&num=1"
        )

        print(f"🔗 Final API URL: {url}")
        response = requests.get(url)
        print(f"🔄 Google CSE Response Code: {response.status_code}")

        # Step 4: Check if Google returned an error
        if response.status_code != 200:
            print(f"❌ Google API error response: {response.text}")
            raise Exception("Google API returned error")

        data = response.json()
        if "items" not in data or len(data["items"]) == 0:
            print("⚠️ No image results found from Google.")
            raise Exception("No image results returned")

        # Step 5: Extract image URL
        image_url = data["items"][0]["link"]
        print(f"🖼️ Image URL: {image_url}")

        # Step 6: Download the image itself
        image_response = requests.get(image_url, timeout=10)
        if image_response.status_code != 200:
            print(f"❌ Failed to download image: {image_response.status_code}")
            raise Exception("Image download failed")

        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        print("✅ Image loaded successfully.")
        return image

    except Exception as e:
        print(f"⚠️ Error in fetch_trend_image: {e}")
        try:
            # Step 7: Fallback to default image if available
            return Image.open("static/default_dress.jpg").convert("RGB")
        except Exception as fallback_error:
            print(f"❌ Could not load fallback image: {fallback_error}")
            raise RuntimeError("Fatal: No image could be loaded.")


# ----------------------------
# Utility: Color + Embedding
# ----------------------------
def extract_dominant_colors(image, k=5, exclude_white=True):
    image = image.resize((150, 150))
    pixels = np.array(image).reshape(-1, 3)
    if exclude_white:
        mask = ~((pixels > 240).all(axis=1) | (pixels < 15).all(axis=1))
        pixels = pixels[mask]
    kmeans = KMeans(n_clusters=k, n_init=10).fit(pixels)
    return kmeans.cluster_centers_

def color_coherence_score(dominant_colors, palette):
    distances = []
    for color in dominant_colors:
        min_dist = np.min(np.linalg.norm(palette - color, axis=1))
        distances.append(min_dist)
    return round(max(0, 100 - np.mean(distances)), 2)

def get_text_embeddings(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        return model.get_text_features(**inputs)

# ----------------------------
# Main Route: Upload + Trend Match
# ----------------------------
@app.route('/trend-match', methods=['POST'])
def trend_match():
    try:
        file = request.files['file']
        user_image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Step 1: Get dynamic fashion trends
        seed_keywords = [
            "satin midi dress", "vintage denim outfit",
            "boho summer look", "leather trench coat", "oversized blazer"
        ]  # Keep to 5 max

        vibe_prompts = fetch_trending_fashion_terms(seed_keywords)

        # Step 2: Get embeddings for trend phrases
        vibe_text_emb = get_text_embeddings(vibe_prompts)

        # Step 3: Embed user-uploaded outfit
        inputs = processor(images=user_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)

        # Step 4: Compare user outfit vs trends
        vibe_scores = torch.nn.functional.cosine_similarity(image_emb, vibe_text_emb)
        top_vibe_idx = vibe_scores.argmax().item()
        top_trend = vibe_prompts[top_vibe_idx]
        vibe_score = vibe_scores[top_vibe_idx].item()

        # Step 5: Get image for that trend → extract palette
        trend_image = fetch_trend_image(top_trend)
        trend_palette = extract_dominant_colors(trend_image)

        # Step 6: Extract user image palette → compute color coherence
        user_palette = extract_dominant_colors(user_image)
        color_score = color_coherence_score(user_palette, trend_palette)

        # Step 7: Combine visual & semantic scores
        match_score = round(vibe_score * 100, 2)
        final_score = round((0.5 * match_score + 0.5 * color_score), 2)

        return jsonify({
            "vibe_trend": top_trend,
            "vibe_score": round(vibe_score, 2),
            "trend_match_score": match_score,
            "color_coherence_score": color_score,
            "final_weighted_score": final_score
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # <--- add this to show the real error in terminal

        return jsonify({"error": str(e)}), 500

# ----------------------------
# Web Form (Optional UI)
# ----------------------------
@app.route('/')
def form():
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
