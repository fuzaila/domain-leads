# Domain Lead Scorer

A Streamlit web application to find potential leads for domain names using SerpAPI and Google's Gemini AI.

## Features
- Extracts keywords from input domains.
- Fetches Google Search results via SerpAPI.
- Scores potential leads using Gemini AI.
- Allows bulk domain processing.
- Caches SerpAPI results to save credits.

## Deployment
This app is intended for deployment on Streamlit Community Cloud.
Required secrets:
- `SERPAPI_KEY`
- `GOOGLE_API_KEY`
