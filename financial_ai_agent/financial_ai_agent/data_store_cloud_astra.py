import os
from langchain_astradb import AstraDBVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from typing import List

import logging
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv("financial_ai_agent/.env")
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cuda:0"}  # Or "cpu" if GPU isn't available
)

# Astra DB setup
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ENDPOINT =  os.getenv("ASTRA_DB_ENDPOINT")

GEMINI_API_KEY =  os.getenv("GEMINI_API_KEY")

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",api_key = GEMINI_API_KEY)

class DataProcessor:
    def __init__(self):
        # Initialize AstraDB collections
        self.summary_store = AstraDBVectorStore(
            collection_name="summaries_index",
            embedding=embeddings,
            api_endpoint=ASTRA_DB_ENDPOINT,
            token=ASTRA_DB_TOKEN,
        )

        self.chunk_store = AstraDBVectorStore(
            collection_name="full_context_index",
            embedding=embeddings,
            api_endpoint=ASTRA_DB_ENDPOINT,
            token=ASTRA_DB_TOKEN,
        )

    def generate_summary(self,content):
        try:
            prompt = f"""Generate a concise summary that preserves key entities and relationships.
            Focus on maintaining searchability for information retrieval purposes.
            Your output should be only the summary. No heading or prefix is expected.
            Content: {content}.
            Summary:\n"""  
            return LLM.invoke(prompt).content
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {str(e)}")
            raise
        

    def process_json_files(self,json_files: List[str]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            add_start_index=True
        )

        for file_path in json_files:
            try:
                # File reading and JSON parsing
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON in {file_path}: {str(e)}")
                        continue

                    # Validate URL exists in data
                    if 'url' not in data or not data['url']:
                        logger.warning(f"⚠️ Skipping file {file_path} - Missing URL")
                        continue
                    
                    url = data['url']
                    metadata = {k: v for k, v in data.items() if k != "content"}

                    # Check if URL already exists
                    try:
                        if self.url_exists(url):
                            logger.info(f"⚠️ Skipping existing URL: {url}")
                            continue
                    except Exception as e:
                        logger.error(f"❌ URL check failed for {url}: {str(e)}")
                        continue

                    # Process summary
                    try:
                        summary = self.generate_summary(data["content"])
                        summary_doc = Document(
                            page_content=summary,
                            metadata=metadata
                        )
                        # Store summary
                        self.summary_store.add_documents([summary_doc])
                        logger.info(f"✅ Added summary for {url}")
                    except Exception as e:
                        logger.error(f"❌ Failed to process summary for {url}: {str(e)}")
                        continue  # Skip chunks if summary fails

                    # Process chunks
                    try:
                        chunks = text_splitter.split_text(data["content"])
                        chunk_docs = []
                        for i, chunk in enumerate(chunks):
                            chunk_meta = metadata.copy()
                            chunk_meta.update({"chunk_seq": i})
                            chunk_docs.append(Document(
                                page_content=chunk,
                                metadata=chunk_meta
                            ))
                        # Store chunks
                        self.chunk_store.add_documents(chunk_docs)
                        logger.info(f"✅ Added {len(chunk_docs)} chunks for {url}")
                    except Exception as e:
                        logger.error(f"❌ Failed to process chunks for {url}: {str(e)}")
                        # Consider rolling back summary here if needed

            except IOError as e:
                logger.error(f"❌ File access error {file_path}: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Unexpected error processing {file_path}: {str(e)}")

    def url_exists(self,url: str) -> bool:
        """Check if URL exists in either summary or chunk stores"""
        try:
            # Check summary store
            existing_summaries = self.summary_store.similarity_search(
                query=" ",  # Dummy query
                k=1,
                filter={"url": url}
            )
            if existing_summaries:
                return True

            # Check chunk store (optional)
            existing_chunks = self.chunk_store.similarity_search(
                query=" ",
                k=1,
                filter={"url": url}
            )
            return len(existing_chunks) > 0
        except Exception as e:
            logger.error(f"❌ Database query failed: {str(e)}")
            raise  # Re-raise to handle in calling function


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]  # Go two levels up from json_upload.py
    json_folder = BASE_DIR / "data/json/"
    json_files_list = [json_folder / i for i in os.listdir(json_folder)]
    data_processor = DataProcessor()
    data_processor.process_json_files([json_files_list[0]])
