import streamlit as st
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
# Import types and tools from your financial AI agent
from financial_ai_agent.agent.types import RAGState, SearchResult, Resource, ResourceType, Error, SearchRequest
from financial_ai_agent.agent.search_tools import SearchFullContentTool, SearchSummaryTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.messages.tool import ToolCall
import torch
import os
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import asyncio


st.set_page_config(
        page_title="Financial Research Assistant",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
st.markdown("""
    <style>
        .thinking-stream {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            border-left: 3px solid #4682B4;
        }
        .source-box {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid #4caf50;
        }
        .main-header {
            color: #1E3A8A;
            font-size: 2.2rem;
        }
        .step-header {
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)




def init_session_state():
    if "action_history" not in st.session_state:
        st.session_state.action_history = []
    if "thinking_history" not in st.session_state:
        st.session_state.thinking_history = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "Deep Search"
    if "embeddings_loaded" not in st.session_state:
        st.session_state.embeddings_loaded = False
    if "search_tools" not in st.session_state:
        st.session_state.search_tools = {}
    if "rag_state" not in st.session_state:
        st.session_state.rag_state = RAGState(question="")
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True
    if "app" not in st.session_state:
        st.session_state.app = None
    if "simple_rag_chain" not in st.session_state:
        st.session_state.simple_rag_chain = None
    if "max_research_steps" not in st.session_state:
        st.session_state.max_research_steps = 3
    if "sidebar_updated" not in st.session_state:
        st.session_state.sidebar_updated = False
    if "answer" not in st.session_state:
        st.session_state.answer = None
    if "sources" not in st.session_state:
        st.session_state.sources = None

# Call initialization
init_session_state()

# Load necessary resources
@st.cache_resource
def load_embeddings_and_tools():
    """Load embeddings and search tools - cached to prevent reloading"""
    from financial_ai_agent.agent.research_agent import app , simple_rag_chain
    from financial_ai_agent.agent.search_tools import SearchSummaryTool, SearchFullContentTool
    
    # Create search tools
    simple_search_tool = SearchFullContentTool("simple_search")
    summary_search_tool = SearchSummaryTool("summary_search")
    
    # Initialize the chain
    
    
    return {
        "app": app,
        "simple_rag_chain": simple_rag_chain,
        "simple_search_tool": simple_search_tool,
        "summary_search_tool": summary_search_tool
    }
# Load resources
with st.spinner("Loading search resources..."):
    if not st.session_state.embeddings_loaded:
        try:
            resources = load_embeddings_and_tools()
            st.session_state.app = resources["app"]
            st.session_state.simple_rag_chain = resources["simple_rag_chain"]
            st.session_state.search_tools["simple"] = resources["simple_search_tool"]
            st.session_state.search_tools["summary"] = resources["summary_search_tool"]
            st.session_state.embeddings_loaded = True
        except Exception as e:
            st.error(f"Error loading resources: {str(e)}")
            st.stop()

# App layout: Title and description
st.markdown("<h1 class='main-header'>Financial Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("""
This AI-powered research assistant can help you find information from financial documents.
Ask any question about financial topics, and the assistant will search through relevant documents to find answers.
""")
# Sidebar controls
def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        if st.button("Refresh Sidebar"):
            st.session_state.sidebar_updated = True
        # Search mode toggle
        search_mode = st.radio(
            "Select Search Mode",
            ["Deep Search", "Simple Search"],
            help="Deep Search uses an agent to perform multiple searches for comprehensive answers. Simple Search performs a single direct search."
        )
        st.session_state.search_mode = search_mode
        
        # Option to clear chat history
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.thinking_history = []
            st.session_state.action_history = []
            st.session_state.rag_state = RAGState(question="")
            st.success("Chat history cleared!")
        st.session_state.max_research_steps = st.slider("Max Research Steps", 1, 10, 3)
        
            

# Always call this, but it conditionally updates based on state
render_sidebar()


# Main content area - Chat display
chat_container = st.container()
with chat_container:
    # Display prior messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("View Sources", expanded=False):
                    for idx, source in enumerate(message["sources"]):
                        resource_url = source.get('resource_url', 'Unknown')
                        resource_type = source.get('resource_type', 'Unknown')
                        text = source.get('text', 'No excerpt available')
                        
                        st.markdown(f"<div class='source-box'>", unsafe_allow_html=True)
                        st.markdown(f"**Source {idx+1}:** [{resource_url}]({resource_url})")
                        st.markdown(f"**Type:** {resource_type}")
                        st.markdown(f"**Excerpt:** {text[:300]}..." if len(text) > 300 else f"**Excerpt:** {text}")
                        st.markdown("</div>", unsafe_allow_html=True)


# Function to stream thinking process
def stream_thinking(text: str):
    if st.session_state.show_thinking:
        thinking_container = st.empty()
        thinking_container.markdown(f"<div class='thinking-stream'>🤔 {text}</div>", unsafe_allow_html=True)
        st.session_state.thinking_history.append(text)
        return thinking_container
    else:
        st.session_state.thinking_history.append(text)
        return st.empty()

async def simple_search(query: str) -> Dict:
    """Perform a direct search using the simple search tool"""
    try:
        
        # Get tool node from session state
        search_tool = st.session_state.search_tools["simple"]
        
        # Direct invocation of search tool
        search_results = await search_tool._arun(
            query=query, 
            limit=5
        )
        simple_rag_chain = st.session_state.simple_rag_chain
        
        chat_history = st.session_state.chat_history[-10:]
    
        llm_answer = await simple_rag_chain(query,search_results,chat_history)

        # Format results
        sources = []
        for result in search_results:
            sources.append({
                "resource_url": result["url"],
                "resource_type": "content",
                "text": result["content"]
            })
        
        return {
            "answer": llm_answer.content,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in simple search: {e}")
        return {
            "answer": f"Sorry, I encountered an error while searching: {str(e)}",
            "sources": []
        }


async def deep_search(question: str):
    """Process a question using the agent-based deep search"""
    # Update RAG state with new question and conversation history
    rag_state = RAGState(
        question=question,
        user_conversation_history=st.session_state.chat_history[-10:],
        research_steps=0,
        max_research_steps=st.session_state.max_research_steps
    )
    
    initial_state = {
            "question": question,
            "structured": rag_state,
        }
    logger.info(f"initial_state: {initial_state}")

    # Original node methods from the app
    app = st.session_state.app

    try:
        # Process the question with patched nodes for streaming updates
        result = await app.ainvoke(initial_state)

        st.session_state.action_history.extend(result['action_history'])
        print(result['action_history'])
        # Update the session RAG state
        st.session_state.rag_state = rag_state
        
        return {
            "answer": result['answer'],
            "sources": result['sources']
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}",
            "sources": []
        }


async def process_question(question: str):
    """Process user question based on selected search mode"""
    # Clear previous thinking history for new questions
    st.session_state.thinking_history = []
    
    # Show initial thinking message
    thinking = stream_thinking(f"Processing your question: '{question}'")
    
    try:
        if st.session_state.search_mode == "Simple Search":
            # Perform simple direct search
            thinking = stream_thinking("Performing direct search on document chunks...")
            result = await simple_search(question)
        else:
            logger.info(f"question:{question}")
            # Perform deep search with agent
            thinking = stream_thinking("Initiating deep search process...")
            result = await deep_search(question)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        result = {
            "answer": f"I encountered an error while processing your question: {str(e)}",
            "sources": []
        }
    finally:
        # Clear thinking indicator
        if thinking:
            thinking.empty()
    
    return result

user_input = st.chat_input("Ask a financial question...")
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "human", "content": user_input})
    
    # Display user message
    with st.chat_message("human"):
        st.markdown(user_input)
    
    # Display assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            # Process the question asynchronously
            result = asyncio.run(process_question(user_input))
            # result = run_async_task(process_question, user_input)

            
            # Extract answer and sources
            st.session_state.answer = result.get("answer", "I couldn't find an answer to your question.")
            st.session_state.sources = result.get("sources", [])
            
            print("SOURCES____",st.session_state.sources)
            # Display the answer
            st.markdown(st.session_state.answer)
            
        # Display sources if available
        if st.session_state.sources:
            with st.expander("View Sources", expanded=False):
                st.session_state.show_thinking = True
            
                for idx, source in enumerate(st.session_state.sources):
                    resource_url = source.get('resource_url', 'Unknown')
                    print("__resource_url__",resource_url)
                    resource_type = source.get('resource_type', 'Unknown')
                    text = source.get('text', 'No excerpt available')
                    
                    st.markdown(f"<div class='source-box'>", unsafe_allow_html=True)
                    st.markdown(f"**Source {idx+1}:** [{resource_url}]({resource_url})")
                    st.markdown(f"**Type:** {resource_type}")
                    st.markdown(f"**Excerpt:** {text[:300]}..." if len(text) > 300 else f"**Excerpt:** {text}")
                    st.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.show_thinking and len(st.session_state.rag_state.action_history)>0:
            # st.header("Agent Thinking History")
            with st.expander("Agent Thinking History", expanded=False):
                for idx, thought in enumerate(st.session_state.rag_state.action_history):
                    st.markdown(f"**Step {idx+1}:**")
                    if thought['role'] == 'assistant' and 'is_final_answer' not in thought.keys():
                        try:
                            thought = {"role":thought['role'],"content": SimpleJsonOutputParser().invoke(thought['content'])}
                        except Exception as e:
                            thought = {"role":thought['role'],"content": thought['content']}
                    st.json(thought,expanded=False)


                
        # Add assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": st.session_state.answer,
            "timestamp": datetime.now().isoformat()
        })



    

