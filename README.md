# App Insights

App Insights is a web application that allows users to gather, analyze, and gain insights from Google Play Store reviews for various apps. By leveraging sentiment analysis and AI-powered insights, it helps identify key pain points, usability issues, and potential areas for improvement in apps.

## Features

- **Search Apps by Topic**: Easily search for apps related to a specific topic or keyword on the Google Play Store.
- **Review Extraction**: Fetch and display user reviews for each app to gather feedback.
- **Sentiment Analysis**: Analyze the sentiment of user reviews to gauge user satisfaction and dissatisfaction.
- **AI-Powered Insights**: Generate actionable insights based on the sentiment analysis of user reviews, highlighting key issues and potential improvements.
- **Intuitive User Interface**: A user-friendly interface for displaying insights.

## Technologies Used

- **Flask**: Lightweight Python web framework for building the backend.
- **Google Play Scraper**: To fetch data (apps and reviews) from the Google Play Store.
- **Transformers (HuggingFace)**: For performing sentiment analysis on app reviews.
- **KeyBERT**: For keyword extraction to analyze key topics from reviews.
- **Google Generative AI**: For generating insights based on the sentiment analysis of reviews.

## Installation

To set up the App Insights project locally, follow the steps below.

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/omkargarad78/AppInsights.git
    cd AppInsights
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your API key**:
    - Get a Google API Key and Gemini AI API Key.
    - Update the keys in your `app.py` where indicated:
    ```python
    GEMINI_API_KEY = "your-api-key"
    ```

5. **Run the application**:
    ```bash
    python app.py
    ```

### Usage

- **Search for apps** by entering a topic (e.g., "fitness", "productivity").
- The app will retrieve the top apps related to your topic.
- **View insights** for each app, including sentiment analysis and AI-powered insights based on user reviews.
