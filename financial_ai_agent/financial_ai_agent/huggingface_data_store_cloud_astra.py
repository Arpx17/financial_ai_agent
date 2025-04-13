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

print(ASTRA_DB_ENDPOINT)

class HuggingFaceDataProcessor:
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

    def process_dataframe(self,dataframe):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            add_start_index=True
        )

        for _,row in dataframe.iterrows():
            url = row['index']
            metadata = dict(
                    url = row['index'],
                    document_type = row['document_type'],
                    language = row['language'],
                    domain = row['domain'],
                    expanded_description = row['document_description'],
                    pii_spans = row['pii_spans'],
                    expanded_type = row['expanded_type'],
                    )
            
            try:
                if self.url_exists(url):
                    logger.info(f"⚠️ Skipping existing URL: {url}")
                    continue
            except Exception as e:
                logger.error(f"❌ URL check failed for {url}: {str(e)}")
                continue
                
            try:
                summary = row['expanded_description']
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

            try:
                chunks = text_splitter.split_text(row['generated_text'])
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
    import pandas as pd

    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/gretelai/synthetic_pii_finance_multilingual/" + splits["train"])
    data_processor = HuggingFaceDataProcessor()
    data_processor.process_dataframe(df)
