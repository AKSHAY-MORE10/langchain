import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import google.generativeai as genai
import os
from  dotenv import load_dotenv

load_dotenv()

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Gemini API Key (replace with your own)
genai.configure(api_key=os.environ.get("AIzaSyDfIgGcRar_Xz3T7_mUMBxGkuFed3G5pW0")) #or "YOUR_API_KEY")

def extract_text_from_website(url):
    """Extracts text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def clean_text(text):
    """Cleans the extracted text by removing noise."""
    if text is None:
        return ""

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def analyze_website_data(text, url):
    """Analyzes the cleaned text using Gemini and corrects data."""
    if not text:
        return "Website data could not be processed."

    model = genai.GenerativeModel('gemini-pro') #or gemini-ultra if available

    prompt = f"""
    Analyze the following website text from {url} and provide:
    1. A summary of the website's content.
    2. Any factual errors or inconsistencies you find.
    3. Corrected versions of the errors, if possible.
    4. Important information about the website.
    5. key words from the site.

    Website Text:
    {text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error analyzing website data."

def run_ai_agent(url):
    """Runs the AI agent to analyze and correct website data."""
    raw_text = extract_text_from_website(url)
    cleaned_text = clean_text(raw_text)
    analysis = analyze_website_data(cleaned_text, url)
    return analysis

# Example usage:
website_url = "https://www.langchain.com/"  # Replace with the website URL you want to analyze.
result = run_ai_agent(website_url)
print(result)

# Install necessary packages (run these commands in your terminal):
# pip install requests beautifulsoup4 nltk google-generativeai