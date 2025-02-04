from flask import Flask, render_template, request, jsonify
from google_play_scraper import search, reviews, app as scrape_app
from transformers import pipeline
from keybert import KeyBERT
import google.generativeai as genai
import os

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini AI API
GEMINI_API_KEY = "your api key"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Function to search apps based on a topic
def search_apps_by_topic(topic):
    try:
        search_results = search(topic, lang="en", country="us")
        app_ids = []
        for result in search_results:
            app_id = result["appId"]
            try:
                app_details = scrape_app(app_id, lang="en", country="us")
                installs = app_details.get("installs", "0").replace("+", "").replace(",", "")
                installs = int(installs) if installs.isdigit() else 0
                rating = app_details.get("score", 0)
                app_ids.append({
                    "appId": app_id,
                    "name": app_details.get("title"),
                    "installs": installs,
                    "rating": rating,
                })
            except Exception as e:
                print(f"Error fetching app details for {app_id}: {e}")
                continue
        sorted_apps = sorted(app_ids, key=lambda x: (x["installs"], x["rating"]), reverse=True)
        return sorted_apps[:3]
    except Exception as e:
        print(f"Error searching apps: {e}")
        return []

# Function to extract reviews
def extract_reviews(app_id, num_reviews=100):
    result, _ = reviews(app_id, lang="en", country="us", count=num_reviews)
    return [r["content"] for r in result if isinstance(r["content"], str) and r["content"].strip()]



def analyze_sentiment(reviews):
    sentiment_model = pipeline("sentiment-analysis")
    analysis_results = []
    for review in reviews:
        if not review or not isinstance(review, str) or review.strip() == "":
            continue  # Skip invalid reviews
        
        try:
            sentiment = sentiment_model(review)[0]
            analysis_results.append({"review": review, "sentiment": sentiment["label"]})
        except Exception as e:
            print(f"Error analyzing sentiment for review: {review}. Error: {e}")
            continue
    
    return analysis_results


def generate_ai_insights(sentiments):
    # Convert sentiments to a readable string
    sentiment_str = "\n".join([f"- {s['review']}: {s['sentiment']}" for s in sentiments])
    
    # Create the prompt
    prompt = f"""
    Based on the following app review sentiments, generate a single, concise paragraph (max 1000 characters) that summarizes key user issues, frustrations, and improvement opportunities. Do not list any points or headings; the response should be in paragraph form. Include insights on:

    - Usability problems such as navigation issues or confusing interface
    - Performance problems like lags, crashes, battery drain, or high memory usage
    - Missing or buggy features that users have requested
    - General user satisfaction and frustration areas
    - Actionable recommendations for improving the app, prioritizing what matters most to users.

    Ensure the paragraph is clear, concise, and covers all the critical areas in a seamless flow without breaking it into separate points. 
    {sentiment_str}
    """
    
    try:
        # Generate AI insights
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text.strip():
            return response.text.strip()
        else:
            return "No AI insights available. The insights could not be generated."
    except Exception as e:
        print(f"Error generating AI insights: {e}")
        return "Failed to generate AI insights. Please try again later."


@app.route("/generate_insights", methods=["POST"])
def generate_insights():
    topic = request.form.get("topic")
    if not topic:
        return jsonify({"error": "Topic is required!"}), 400
    
    print(f"Searching apps for topic: {topic}")
    app_ids = search_apps_by_topic(topic)
    if not app_ids:
        return jsonify({"error": "No apps found for the given topic."}), 404
    
    insights = {}
    for app_data in app_ids:
        app_id = app_data["appId"]
        print(f"Fetching reviews for app: {app_id}")
        
        reviews_texts = extract_reviews(app_id, num_reviews=100)
        if not reviews_texts:
            insights[app_id] = {
                "app_name": app_data["name"],
                "downloads": app_data["installs"],
                "rating": round(app_data["rating"], 2),
                "ai_insights": "No reviews available. Encourage users to leave feedback."
            }
            continue
        
        print(f"Analyzing sentiment for app: {app_id}")
        sentiments = analyze_sentiment(reviews_texts)
        
        print(f"Generating AI insights for app: {app_id}")
        ai_insights = generate_ai_insights(sentiments)
        
        insights[app_id] = {
            "app_name": app_data["name"],
            "downloads": app_data["installs"],
            "rating": round(app_data["rating"], 2),
            "ai_insights": ai_insights
        }
    
    return render_template("results.html", insights=insights)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use dynamic port assignment
    app.run(host="0.0.0.0", port=port)
