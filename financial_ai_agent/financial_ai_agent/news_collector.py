from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from pymongo import MongoClient
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import dateparser
import json
import logging
import os
from newsdataapi import NewsDataApiClient
import requests
import hashlib
import time
from dotenv import load_dotenv
load_dotenv("financial_ai_agent/.env")

logger = logging.getLogger(__name__)

api_key = os.getenv("GEMINI_API_KEY")
api_key_newsdata = os.getenv("NEWSDATA_API_KEY")

# Initialize components
api = NewsDataApiClient(apikey=api_key_newsdata)  

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["news_db"]
collection = db["articles"]

# Step 2: News search
def fetch_news_links(query):
    response = api.latest_api(q=query,country="us", language = "en",)    
    return response['results']

def batched_fetch_news_links(query, total=100, batch_size=10):
    articles = []
    seen_urls = set()

    for _ in range(total // batch_size):
        logger.info(f"Fetching batch {_+1} of {total // batch_size} articles...")
        print((f"Fetching batch {_+1} of {total // batch_size} articles..."))
        batch = fetch_news_links(query)
        print(len(batch))
        print("sleeping for 1 minute")
        time.sleep(60)
    
        for article in batch:
            url = article.get("link")
            if url and url not in seen_urls:
                articles.append(article)
                seen_urls.add(url)
        
        if len(articles) >= total:
            break

    return articles


def extract_info_from_html(source_url: str) -> dict:
    html_content = requests.get(source_url).text
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract title
    title = soup.title.string.strip() if soup.title else None

    def get_meta_content(attrs):
        tag = soup.find("meta", attrs=attrs)
        return tag.get("content") if tag and tag.has_attr("content") else None

    author = (
        get_meta_content({"name": "author"}) or 
        get_meta_content({"property": "article:author"})
    )
    
    date_raw = (
        get_meta_content({"property": "article:published_time"}) or 
        get_meta_content({"name": "date"}) or 
        get_meta_content({"itemprop": "datePublished"})
    )
    date_of_publish = dateparser.parse(date_raw).isoformat() if date_raw else None

    site_name = get_meta_content({"property": "og:site_name"})
    tags = soup.find_all("meta", attrs={"property": "article:tag"})
    tags = [tag.get("content") for tag in tags if tag.has_attr("content")]

    # Clean HTML
    for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "img", "figure", "a", "iframe", "video", "audio", "source"]):
        element.decompose()

    # Extract main content
    article_tag = soup.find("article")
    text = article_tag.get_text(separator=" ", strip=True) if article_tag else soup.get_text(separator=" ", strip=True)
    content = ' '.join(text.split())

    # Fallback site_name
    if not site_name and source_url:
        site_name = urlparse(source_url).netloc.replace("www.", "")

    result = {
        "url": source_url,
        "title": title,
        "date_of_publish": date_of_publish,
        "author": author,
        "site_name": site_name,
        "tags": tags or [],
        "content": content
    }

    # === Saving logic ===
    # Create folders if they don’t exist
    os.makedirs("data/json", exist_ok=True)
    os.makedirs("data/text", exist_ok=True)

    # Unique filename from URL hash
    hash_name = hashlib.sha1(source_url.encode()).hexdigest()[:10]  # Short unique name

    # Save JSON
    json_path = os.path.join("data/json", f"{hash_name}.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(result, f_json, ensure_ascii=False, indent=2)

    # Save plain text
    txt_path = os.path.join("data/text", f"{hash_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(content)

    return result

# Step 4: MongoDB storage
def store_in_mongodb(doc):
    existing = collection.find_one({"url": doc["url"]})
    if not existing:
        collection.insert_one(doc)
        logger.info(f"✅ Stored: {doc['title']}")
    else:
        logger.info(f"⚠️ Already exists: {doc['title']}")
        
batched_fetch_chain = RunnableLambda(lambda query: batched_fetch_news_links(query))
extractor_chain = RunnableLambda(lambda articles: [extract_info_from_html(a["link"]) for a in articles])
storage_chain = RunnableLambda(lambda docs: [store_in_mongodb(doc) for doc in docs])

full_news_ingestion_chain = (
    batched_fetch_chain 
    | extractor_chain 
    | storage_chain
)

query = "financial OR business OR finance"

# Run the pipeline
full_news_ingestion_chain.invoke(query)
