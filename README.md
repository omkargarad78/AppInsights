# App Insights

App Insights is a web application that allows users to gather, analyze, and gain insights from Google Play Store reviews for various apps. By leveraging sentiment analysis and AI-powered insights, it helps identify key pain points, usability issues, and potential areas for improvement in apps.

## Features

- **Search Apps by Topic**: Easily search for apps related to a specific topic or keyword on the Google Play Store.
- **Review Extraction**: Fetch and display user reviews for each app to gather feedback.
- **Sentiment Analysis**: Analyze the sentiment of user reviews to gauge user satisfaction and dissatisfaction.
- **AI-Powered Insights**: Generate actionable insights based on the sentiment analysis of user reviews, highlighting key issues and potential improvements.
- **Intuitive User Interface**: A user-friendly interface for displaying insights.

## Technologies Used

- **FastAPI**: A modern, high-performance web framework for building APIs with Python- **Google Play Scraper**: To fetch data (apps and reviews) from the Google Play Store.
- **Transformers (HuggingFace)**: For performing sentiment analysis on app reviews.
- **KeyBERT**: For keyword extraction to analyze key topics from reviews.
- **Google Generative AI (Gemini)**: For generating insights based on the sentiment analysis of reviews.

## Installation

To set up the App Insights project locally, follow the steps below.

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AppInsights.git
   cd AppInsights
