from langchain_core.tools import BaseTool
from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import os
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
from financial_ai_agent.agent.types import SearchResult
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

summary_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="summaries_index",
            token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT"),
        )

content_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="full_context_index",
            token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT"),
        )
class SearchSummaryTool(BaseTool):
    name: str = "search_summary"
    description: str = (
        "Perform a similarity search over a collection of summaries generated from the full articles. " \
        "This tool is useful for searching through index when you you are not sure about what you should search. This is meant for starting of your search"
    )
    def __init__(self,tool_name):
        super().__init__()
        """Initialize summary vector stores - one for summaries """
        self.name = tool_name
    
        
    def _run(self, query: str, limit: int = 5, urls: Optional[List[str]] = None):
        """Search in the summary collection"""
        filter_dict = {"url": {"$in": urls}} if urls else None
        docs_with_scores = summary_store.asimilarity_search_with_score(
            query=query,
            k=limit,
            filter=filter_dict
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append(SearchResult(
                content=doc.page_content,
                url=doc.metadata["url"],
                score=float(score),
                metadata=doc.metadata
            ).model_dump())
        
        return results
    async def _arun(self, query: str, limit: int = 5, urls: Optional[List[str]] = None):
        """Search in the summary collection"""
        filter_dict = {"url": {"$in": urls}} if urls else None
        docs_with_scores = await summary_store.asimilarity_search_with_score(
            query=query,
            k=limit,
            filter=filter_dict
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append(SearchResult(
                content=doc.page_content,
                url=doc.metadata["url"],
                score=float(score),
                metadata=doc.metadata
            ).model_dump())
        
        return results

class SearchFullContentTool(BaseTool):
    name: str = "search_full_content"
    description: str = (
        "Perform a similarity search over a collection of chunked texts generated from the full articles. " \
        "This tool is useful for searching through index when you have not enough information to answer on a topic or question"
    )    
    def __init__(self,tool_name):
        super().__init__()
        """Initialize summary vector stores - one for summaries """
        self.name = tool_name

    def _run(self, query: str, limit: int = 5, urls: Optional[List[str]] = None):
        """Search in the content chunks collection"""
        filter_dict = {"url": {"$in": urls}} if urls else None
        docs_with_scores = content_store.asimilarity_search_with_score(
            query=query,
            k=limit,
            filter=filter_dict
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append(SearchResult(
                content=doc.page_content,
                url=doc.metadata["url"],
                score=float(score),
                metadata=doc.metadata
            ).model_dump())
        
        return results
        
    async def _arun(self, query: str, limit: int = 5, urls: Optional[List[str]] = None) :
        """Search in the content chunks collection"""
        filter_dict = {"url": {"$in": urls}} if urls else None
        docs_with_scores = await content_store.asimilarity_search_with_score(
            query=query,
            k=limit,
            filter=filter_dict
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append(SearchResult(
                content=doc.page_content,
                url=doc.metadata["url"],
                score=float(score),
                metadata=doc.metadata
            ).model_dump())
        
        return results