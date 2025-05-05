from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from keybert import KeyBERT
from google_play_scraper import search, reviews, app as scrape_app
import uvicorn
import asyncio
import functools
import concurrent.futures
from typing import List, Dict, Any, Tuple
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = FastAPI()


templates = Jinja2Templates(directory="templates")

# =============== Startup Event ===============
@app.on_event("startup")
async def load_models():
    print("[DEBUG] Loading models... please wait...")
    # Use a smaller, faster model for sentiment analysis
    app.state.cuda_available = False
    try:
        import torch
        app.state.cuda_available = torch.cuda.is_available()
        print(f"[DEBUG] CUDA available: {app.state.cuda_available}")
    except ImportError:
        print("[DEBUG] PyTorch not installed, using CPU only")
        
    app.state.sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if app.state.cuda_available else -1  # Use GPU if available
    )
    app.state.kw_model = KeyBERT()
    # Create a thread pool for CPU-bound tasks
    app.state.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    # Create a simple cache for app details and reviews
    app.state.app_cache = {}
    app.state.review_cache = {}
    
    # Configure Gemini API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("[WARNING] GEMINI_API_KEY not found in environment variables. Gemini features will be disabled.")
        app.state.gemini_available = False
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            app.state.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            app.state.gemini_available = True
            print("[DEBUG] Gemini API configured successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to configure Gemini API: {e}")
            app.state.gemini_available = False
    
    print("[DEBUG] Models loaded successfully!")

# =============== Functions ===============

# Search apps by topic with caching
def search_apps_by_topic(topic):
    print(f"[DEBUG] Searching for apps with topic: {topic}")
    try:
        # Check cache first
        if topic in app.state.app_cache:
            print(f"[DEBUG] Using cached results for topic: {topic}")
            return app.state.app_cache[topic]
            
        search_results = search(topic, lang="en", country="in")
        app_ids = []
        
        # Process only the top 5 results to speed up initial search
        for result in search_results[:4]:
            app_id = result["appId"]
            try:
                app_details = scrape_app(app_id, lang="en", country="us")
                installs = app_details.get("installs", "0").replace("+", "").replace(",", "")
                installs = int(installs) if installs.isdigit() else 0
                rating = app_details.get("score") or 0.0
                app_ids.append({
                    "appId": app_id,
                    "name": app_details.get("title"),
                    "installs": installs,
                    "rating": rating,
                })
                print(f"[DEBUG] Fetched details for: {app_id}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch details for {app_id}: {e}")
                
        sorted_apps = sorted(app_ids, key=lambda x: (x["installs"], x["rating"]), reverse=True)
        result = sorted_apps[:3]
        
        # Cache the results
        app.state.app_cache[topic] = result
        return result
    except Exception as e:
        print(f"[ERROR] Failed to search apps: {e}")
        return []

# Asynchronous review extractor with caching
async def extract_reviews_async(app_id, num_reviews=500):  
    print("------")
    print(f"[DEBUG] Extracting reviews for: {app_id}")
    
    # Check cache first
    if app_id in app.state.review_cache:
        print(f"[DEBUG] Using cached reviews for: {app_id}")
        return app_id, app.state.review_cache[app_id]
    
    try:
        result, _ = reviews(app_id, lang='en', country='us', count=num_reviews)
        review_texts = [
            review["content"]
            for review in result
            if review.get("content") and isinstance(review["content"], str) and review["content"].strip() != ""
        ]
        print(f"[DEBUG] Reviews extracted for {app_id}: {len(review_texts)} reviews")
        print("-----")
        
        # Cache the results
        app.state.review_cache[app_id] = review_texts
        return app_id, review_texts
    except Exception as e:
        print(f"[ERROR] Error fetching reviews for {app_id}: {e}")
        return app_id, []

# Process reviews in batches for sentiment analysis
def analyze_sentiment(reviews_texts, app):
    print(f"[DEBUG] Performing sentiment analysis")
    analysis_results = []
    sentiment_model = app.state.sentiment_model
    
    # Process in batches to improve performance
    batch_size = 16  # Adjust based on your available memory
    
    # Filter out invalid reviews first
    valid_reviews = [review for review in reviews_texts 
                    if review and isinstance(review, str) and review.strip() != ""]
    
    # Process in batches
    for i in range(0, len(valid_reviews), batch_size):
        batch = valid_reviews[i:i+batch_size]
        try:
            # Process the entire batch at once
            sentiments = sentiment_model(batch)
            for j, sentiment in enumerate(sentiments):
                analysis_results.append({
                    "review": batch[j],
                    "sentiment": sentiment["label"],
                    "score": sentiment["score"]
                })
        except Exception as e:
            print(f"[ERROR] Batch sentiment analysis failed: {e}")
    
    return analysis_results

# Use Gemini API to analyze pain points and provide business improvement suggestions
async def analyze_with_gemini(app_name, reviews, app):
    print(f"[DEBUG] Analyzing with Gemini API for {app_name}")
    
    if not app.state.gemini_available:
        print("[WARNING] Gemini API not available, falling back to keyword extraction")
        return await extract_pain_points(reviews, app)
    
    # Use only negative reviews for pain point extraction
    negative_reviews = [r["review"] for r in reviews if r["sentiment"] == "NEGATIVE"]
    
    # If we have too many negative reviews, sample them
    if len(negative_reviews) > 20:
        import random
        negative_reviews = random.sample(negative_reviews, 20)
    
    # Prepare the prompt for Gemini
    reviews_text = "\n\n".join(negative_reviews[:10])  # Limit to 10 reviews to avoid token limits
    
    prompt = f"""
    You are an expert app analyst and business consultant. I will provide you with negative reviews for the app '{app_name}'.
    
    Based on these reviews, please:
    1. Identify the top 2-3 pain points users are experiencing
    2. For each pain point, provide a detailed and specific business improvement suggestion that is tailored to this exact app and its unique issues
    3. Your solutions should be actionable, specific, and provide strategic business advice
    4. Format your response as a JSON array with objects containing 'pain_point' and 'solution' keys
    
    Here are the reviews:
    {reviews_text}
    
    Return ONLY a valid JSON array without any additional text or explanation.
    Example format:
    [
      {{
        "pain_point": "Slow loading times",
        "solution": "Optimize database queries and implement caching to reduce loading times by at least 50%. Consider using a CDN for static assets."
      }},
      {{
        "pain_point": "Frequent crashes",
        "solution": "Implement better error handling and crash reporting. Allocate development resources to fix the top 3 crash scenarios identified in analytics."
      }}
    ]
    """
    
    try:
        response = await asyncio.to_thread(
            lambda: app.state.gemini_model.generate_content(prompt).text
        )
        
        # Clean up the response to ensure it's valid JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        import json
        try:
            pain_points = json.loads(response)
            print(f"[DEBUG] Gemini identified {len(pain_points)} pain points with custom solutions")
            return pain_points
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse Gemini response as JSON: {e}")
            print(f"Response was: {response[:100]}...")
            # Fall back to keyword extraction
            return await extract_pain_points(reviews, app)
            
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        # Fall back to keyword extraction
        return await extract_pain_points(reviews, app)

# Extract pain points with keyword extraction (fallback method)
async def extract_pain_points(reviews, app):
    print(f"[DEBUG] Extracting pain points from reviews using keyword extraction")
    
    # Try to use Gemini with a simpler prompt if available
    if app.state.gemini_available:
        try:
            # Use only negative reviews
            negative_reviews = [r["review"] for r in reviews if r["sentiment"] == "NEGATIVE"]
            if not negative_reviews and reviews:
                negative_reviews = [r["review"] for r in reviews][:5]  # Use any reviews if no negative ones
                
            if negative_reviews:
                # Create a simple prompt
                reviews_sample = "\n".join(negative_reviews[:5])
                prompt = f"Based on these app reviews, list 3 pain points and solutions in JSON format [{{'pain_point': 'issue', 'solution': 'detailed fix'}}]: {reviews_sample}"
                
                response = await asyncio.to_thread(
                    lambda: app.state.gemini_model.generate_content(prompt).text
                )
                
                # Clean up response
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                import json
                try:
                    pain_points = json.loads(response)
                    if isinstance(pain_points, list) and len(pain_points) > 0:
                        print(f"[DEBUG] Fallback Gemini identified {len(pain_points)} pain points")
                        return pain_points
                except:
                    pass  # Continue to keyword extraction if JSON parsing fails
        except Exception as e:
            print(f"[ERROR] Fallback Gemini analysis failed: {e}")
    
    # If Gemini failed or is not available, use keyword extraction
    all_reviews = [r["review"] for r in reviews]
    
    # Use only negative reviews for pain point extraction
    negative_reviews = [r["review"] for r in reviews if r["sentiment"] == "NEGATIVE"]
    if not negative_reviews and reviews:
        negative_reviews = all_reviews[:10]  # Use any reviews if no negative ones
    
    # If we have too many negative reviews, sample them
    if len(negative_reviews) > 50:
        import random
        negative_reviews = random.sample(negative_reviews, 50)
    
    pain_keywords = []
    kw_model = app.state.kw_model
    
    # Process keywords extraction in parallel
    def extract_keywords_from_review(review):
        try:
            keywords = kw_model.extract_keywords(review, top_n=3)
            return [kw[0] for kw in keywords]
        except Exception as e:
            print(f"[ERROR] Keyword extraction failed: {e}")
            return []
    
    # Use thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        keyword_lists = list(executor.map(extract_keywords_from_review, negative_reviews))
    
    # Flatten the list of lists
    for keywords in keyword_lists:
        pain_keywords.extend(keywords)
    
    pain_keywords = list(set(pain_keywords))

    # Generate dynamic solutions using the keywords as context
    identified_solutions = []
    for keyword in pain_keywords[:5]:  # Limit to top 5 keywords
        identified_solutions.append({
            "pain_point": keyword,
            "solution": f"The app needs improvement in the area of '{keyword}'. Consider analyzing user feedback specifically about this issue and prioritize fixing it in the next update."
        })
    
    print(f"[DEBUG] Identified {len(identified_solutions)} pain points using keyword extraction")
    return identified_solutions

# Format download numbers (already fast, no optimization needed)
def format_downloads(number):
    print("[DEBUG] Formatting download number")
    if number >= 10**7:
        return f"{number // 10**6}M"
    elif number >= 10**5:
        return f"{number // 10**5}L"
    elif number >= 10**3:
        return f"{number // 10**3}k"
    else:
        return str(number)

# =============== Routes ===============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("[DEBUG] Rendering homepage")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate_insights", response_class=HTMLResponse)
async def generate_insights(request: Request, topic: str = Form(...)):
    print("[DEBUG] Generating insights")
    if not topic:
        print("[ERROR] Topic not provided")
        return templates.TemplateResponse("index.html", {"request": request, "error": "Topic is required!"})

    app_ids = search_apps_by_topic(topic)
    if not app_ids:
        return templates.TemplateResponse("index.html", {"request": request, "error": "No apps found!"})

    # Async review fetch
    tasks = [extract_reviews_async(app_data["appId"]) for app_data in app_ids]
    results = await asyncio.gather(*tasks)

    insights = {}
    for app_data in app_ids:
        app_id = app_data["appId"]
        review_data = dict(results).get(app_id, [])
        if not review_data:
            print(f"[DEBUG] No reviews for {app_id}")
            insights[app_id] = {
                "app_name": app_data["name"],
                "downloads": format_downloads(app_data["installs"]),
                "rating": round(app_data["rating"], 2),
                "pain_points": [{"pain_point": "No reviews", "solution": "Encourage feedback by implementing in-app review prompts and offering incentives for user feedback."}],
            }
            continue

        sentiments = analyze_sentiment(review_data, request.app)
        
        # Use Gemini for pain point analysis
        pain_points = await analyze_with_gemini(app_data["name"], sentiments, request.app)

        insights[app_id] = {
            "app_name": app_data["name"],
            "downloads": format_downloads(app_data["installs"]),
            "rating": round(app_data["rating"], 2),
            "pain_points": pain_points or [{"pain_point": "None found", "solution": "Continue monitoring user feedback for emerging issues."}]
        }

    print("[DEBUG] Finished generating insights")
    return templates.TemplateResponse("results.html", {"request": request, "insights": insights})

# =============== Main Server ===============
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
