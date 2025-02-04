import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyBMr1VwnfUCbQ_NV1TsXPPsP1KRf_p5Kts"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

def generate_ai_insights(sentiments):
    # Convert sentiments to a readable string
    sentiment_str = "\n".join([f"- {s['review']}: {s['sentiment']}" for s in sentiments])
    
    # Create the prompt
    prompt = f"""
Based on the following app review sentiments, generate a concise yet comprehensive insight (max 1000 characters) on how the app can be improved. Identify key user issues, what is not working well, and specific problems users face. Ensure no critical points are missed while keeping the response brief. Focus on usability, performance, features, and overall experience. Provide actionable recommendations for improvement.    {sentiment_str}
    """
    
    try:
        # Generate AI insights
        response = model.generate_content(prompt)
        if response and hasattr(response, "text"):
            return response.text.strip()
        else:
            return "No AI insights available."
    except Exception as e:
        print(f"Error generating AI insights: {e}")
        return "Failed to generate AI insights."
    
# Sample sentiments
sample_sentiments = [
    {"review": "I love this app!", "sentiment": "POSITIVE"},
    {"review": "This app crashes all the time.", "sentiment": "NEGATIVE"},
    {"review": "It's okay, but could be better.", "sentiment": "NEUTRAL"},
]

# Generate AI insights
insights = generate_ai_insights(sample_sentiments)
print(insights)