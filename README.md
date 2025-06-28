🧥 Hybrid Fashion Trend Matcher — Style Studio: Game Zone 🎮
A Flask-based AI app that compares a user's fashion design to real-time trending fashion styles using a hybrid of:

🤖 CLIP-based vision-language matching

🎨 Color palette coherence analysis

📈 Live trend data from Google Trends

🔍 Dynamic image search (DuckDuckGo or Google CSE)

This powers the Game Zone feature in Style Studio, where users test how well their outfit matches live fashion trends.

🚀 Features
✅ Upload an outfit image and get a score on how trendy it is.

✅ Uses CLIP model (fashion-clip) to compare image with trending styles.

✅ Extracts dominant color palette and compares with real-trend palettes.

✅ Trends are fetched dynamically via Google Trends.

✅ Trend images are fetched from DuckDuckGo or Google CSE.

✅ Outputs a final trend match score with detailed breakdown.

🧠 Model Architecture
Component	Purpose
fashion-clip	Vision-language model for vibe similarity
KMeans	Color palette extraction
pytrends	Gets trending fashion phrases
requests	Downloads real images from search results

📂 Project Structure
php
Copy
Edit
/Hybrid Model/
│
├── model.py                 # Main Flask app
├── static/
│   └── default_dress.jpg    # Fallback image if search fails
├── templates/
│   └── form.html            # Simple UI to upload image
├── requirements.txt
└── README.md
💡 How It Works
User uploads an outfit image

App:

Fetches top 5 trending fashion terms from Google Trends

Fetches real images of those trends

Extracts color palettes from the trend image + user's image

Compares semantic similarity via CLIP

Scores color coherence

Returns a combined trend match score out of 100.

🖼️ Example Output (JSON)
json
Copy
Edit
{
  "color_coherence_score": 56.15,
  "final_weighted_score": 39.44,
  "trend_match_score": 22.74,
  "vibe_score": 0.23,
  "vibe_trend": "isPartial"
}
🛠️ Setup Instructions
Clone the project:

bash
Copy
Edit
git clone https://github.com/yourusername/style-studio-hybrid.git
cd style-studio-hybrid
Create a virtual environment (optional):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python model.py
Open in browser:
http://localhost:5000

🧪 Requirements
Python 3.10 or 3.11 (3.12 works with direct API requests)

transformers, torch, pytrends, flask, scikit-learn, Pillow, requests

🔒 Notes
Make sure static/default_dress.jpg exists to avoid fallback errors.

You can toggle between DuckDuckGo image fetching or Google CSE depending on API limits.

📦 Use Case: Style Studio – Game Zone 🎮
This model powers a fun feature in Style Studio, where users upload their designs and see how "on-trend" they are based on real-time fashion analysis. Scores, recommendations, and badges make it a gamified and engaging user experience.

📬 Credits
Model: patrickjohncyh/fashion-clip

Trend API: pytrends

Color logic inspired by Google Material Design palette extraction

