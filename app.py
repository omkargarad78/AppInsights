from flask import Flask, render_template, request, jsonify
from google_play_scraper import search, app, reviews
from transformers import pipeline
from keybert import KeyBERT
from google_play_scraper import app as scrape_app

# Initialize Flask app
app = Flask(__name__)

# Function to search apps based on a topic
# Function to search apps based on a topic
def search_apps_by_topic(topic):
    search_results = search(topic, lang="en", country="us")
    app_ids = []
    for result in search_results:
        app_id = result["appId"]
        app_details = scrape_app(app_id, lang="en", country="us")

        # Safely handle the installs value
        installs = app_details.get("installs", "0")
        if installs:
            installs = installs.replace("+", "").replace(",", "")
            try:
                installs = int(installs)
            except ValueError:
                installs = 0
        else:
            installs = 0

        rating = app_details.get("score", 0)
        app_ids.append(
            {
                "appId": app_id,
                "name": app_details.get("title"),
                "installs": installs,
                "rating": rating,
            }
        )

    sorted_apps = sorted(app_ids, key=lambda x: (x["installs"], x["rating"]), reverse=True)
    return sorted_apps[:3] #It will take top 3 apps with higest downloads and users


# Function to extract reviews
def extract_reviews(app_id, num_reviews=10000):
    result, _ = reviews(app_id, lang="en", country="us", count=num_reviews)
    review_texts = [review["content"] for review in result]
    return review_texts

# Function to analyze sentiment
def analyze_sentiment(reviews):
    sentiment_model = pipeline("sentiment-analysis")
    analysis_results = []
    for review in reviews:
        sentiment = sentiment_model(review)[0]
        analysis_results.append(
            {"review": review, "sentiment": sentiment["label"], "score": sentiment["score"]}
        )
    return analysis_results

# Function to extract pain points
def extract_pain_points(reviews):
    kw_model = KeyBERT()
    all_reviews = [r["review"] for r in reviews]
    pain_keywords = []
    for review in all_reviews:
        keywords = kw_model.extract_keywords(review, top_n=3)
        pain_keywords.extend([kw[0] for kw in keywords])
    pain_keywords = list(set(pain_keywords))  # Remove duplicates

    solutions = {
        "crash": "Fix the app stability issues by releasing a new update and testing it on multiple devices.",
        "bug": "Investigate reported bugs and fix them in the next patch.",
        "lag": "Improve performance and optimize the app's response time.",
        "battery drain": "Optimize battery usage by reducing background processes and optimizing app performance.",
        "feature missing": "Consider adding the requested features to the next version of the app.",
        "slow": "Improve app loading speed by optimizing resources and reducing load time.",
        "notifications": "Ensure that notifications are working properly and notify users of important updates.",
        "interface": "Work on improving the user interface by making it more intuitive and easier to navigate.",
        "ads": "Reduce the frequency of ads or provide an option for users to remove them for a better experience.",
    }

    identified_solutions = []
    for keyword in pain_keywords:
        if keyword in solutions:
            identified_solutions.append(
                {"pain_point": keyword, "solution": solutions[keyword]}
            )
    return identified_solutions

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for generating insights
# Helper function to format downloads
def format_downloads(number):
    if number >= 10**7:
        return f"{number // 10**6}M"  # Convert to millions
    elif number >= 10**5:
        return f"{number // 10**5}L"  # Convert to lakhs
    elif number >= 10**3:
        return f"{number // 10**3}k"  # Convert to thousands
    else:
        return str(number)  # Keep as is for smaller numbers

# Updated generate_insights function
@app.route("/generate_insights", methods=["POST"])
def generate_insights():
    topic = request.form.get("topic")
    if not topic:
        return jsonify({"error": "Topic is required!"}), 400

    app_ids = search_apps_by_topic(topic)
    if not app_ids:
        return jsonify({"error": "No apps found for the given topic."}), 404

    insights = {}
    for app_data in app_ids:
        app_id = app_data["appId"]
        reviews_texts = extract_reviews(app_id, num_reviews=100)
        sentiments = analyze_sentiment(reviews_texts)
        pain_points_with_solutions = extract_pain_points(sentiments)

        # Format downloads
        formatted_downloads = format_downloads(app_data["installs"])

        # Format the rating to 2 decimal places
        rating = round(app_data["rating"], 2)

        insights[app_id] = {
            "app_name": app_data["name"],
            "downloads": formatted_downloads,
            "rating": rating,
            "pain_points": pain_points_with_solutions,
        }

    return render_template("results.html", insights=insights)


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
