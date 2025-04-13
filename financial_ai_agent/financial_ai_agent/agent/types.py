# Import necessary libraries
import os
import json
import hashlib
import logging
import sys
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ValidationError



class ResourceType(str, Enum):
    SUMMARY = "summary"
    CONTENT = "content"
    METADATA = "metadata"

class Resource(BaseModel):
    resource_url: Union[str,int]
    resource_type: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    resource_url: Union[str,int]
    resource_type: str
    text: Optional[str] = None

class Error(BaseModel):
    exception: str
    type: str
    detail: str
    data: Optional[Dict] = None

class SearchResult(BaseModel):
    content: str 
    url: Union[str,int]
    score: float
    metadata: Dict[str, Any] = {}

class NodeLabels(str, Enum):
    SELECT_ACTION = "select_action"
    SEARCH_SUMMARY = "search_summary"
    SEARCH_FULL_CONTENT = "search_full_content" 
    GENERATE_ANSWER = "generate_answer"
    REPORT_RESULT = "report_result"

class ActionType(str, Enum):
    ANSWER = "answer"
    SEARCH_SUMMARY = "search_summary" 
    SEARCH_FULL_CONTENT = "search_full_content"


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    urls: Optional[List[Union[str, int]]] = None

class Action(BaseModel):
    action: str
    reason: str
    query: Optional[str] = None
    urls: Optional[List[Union[str, int]]] = None
    answer: Optional[str] = None
    sources: Optional[List[Source]] = None

class RAGState(BaseModel):
    """State for the RAG system"""
    question: str
    user_conversation_history: List[Dict] = []
    research_steps: int = 0
    max_research_steps: int = 10
    errors: List[Error] = []
    summaries: List[SearchResult] = []
    content_chunks: List[SearchResult] = []
    valid_sources: List[Resource] = []
    action_history: List[Dict] = []