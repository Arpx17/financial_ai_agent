from langchain.schema import HumanMessage, AIMessage
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableLambda
)
from pydantic import  ValidationError
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
# Text processing imports
from enum import Enum
from typing import Dict
from financial_ai_agent.agent.types import (
    RAGState,
    Action, 
    ActionType, 
    NodeLabels, 
    SearchRequest, 
    Resource, 
    Error, 
    ResourceType
)
from financial_ai_agent.agent.search_tools import SearchSummaryTool, SearchFullContentTool
import datetime
import os
from langchain_core.messages.tool import ToolCall
from langgraph.prebuilt import ToolNode
import uuid
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import logging
from dotenv import load_dotenv
load_dotenv("financial_ai_agent/.env")

logger = logging.getLogger(__name__)

GEMINI_API_KEY =  os.getenv("GEMINI_API_KEY")

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",api_key = GEMINI_API_KEY)

async def simple_rag_chain(question: str, reference: str, history: str):
    """Build a simple RAG chain for generating answers"""
    # Define the prompt template
    simple_rag_prompt = """
    You are an helpful assistant who understands the user query and provide answers.
    You will be provided with relevent data related to the question which you will use to answer the question.
    Make sure to answer accurately and if you find that you can not answer from the available information then you should clearly mention that.
    Don't try to answer from your own knowledge. 

    You will also prodived with the history of conversations between you and the user. 
    So if the user asks any subsequent question then you should check the history to understand the context better.
    Make sure to use the history of conversation and if you don't find anything in the history as well then also cleary menion that
    """

    messages = {"role":"human",
                "content":[{"type": "text", 
                            "text": f"""## Question: \n\n {question} \n\n 
                                    ## History: \n\n {history} \n\n
                                    ## Reference Documents: \n\n {reference} """
                        },  # Optional: Add a text prompt
                ]
            }

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system",simple_rag_prompt),
                MessagesPlaceholder(variable_name = "messages")
            ]   
        )
    
    chain = prompt | LLM

    return await chain.ainvoke(input = {"messages":[messages]})

SELECT_ACTION_PROMPT = """You are an intelligent research assistant that helps answer questions by searching through relevant information.
For searching you are provided with two search method. One is summary search which gives you a high level overview of the documents and the other is full context search which gives you the detailed content of the documents.
You can start searching from the summaries and then if you find that you need more detailed information then you can search for full context.
You can also insert assertions which are supporting facts relevent to the question and might help to answer the question accurately.
If you feel that you have collected enough information to answer the question then you can also end the process .
You will also be provided with the actions you have taken so far. you should learn from them and understand the issue you if any by yourself. 
The user may ask some follow up questions based on the final answer which is genereated by summarizer llm (NOT YOU). 
So First you have to check the current available context and search results are enough to answer the user question. Otherwise you can search again
For the follow up questions your resopnse structure should be same as the first question.

Your task is to determine the next best action to take in order to answer the user's question.

USER CONVERSATION HISTORY:
{user_history}

QUESTION: {question}

YOUR ACTION HISTORY:
{history}

SUMMARIES FOUND:
{summaries}

DETAILED CONTENT CHUNKS:
{content_chunks}

CURRENT RESEARCH STEP: {research_steps} / {max_research_steps}

Based on the available information, decide the next action to take. You have the following options:

1. Search for summaries to gather high-level relevant information
2. Search for detailed content on specific documents to get more in-depth information
3. Insert assertions which are supporting facts relevent to the question and might help to answer the question accurately
4. End the process if answering is not possible or if the question has been fully addressed

Respond with a JSON object describing your chosen action:

For searching summaries:
```json
{{
  "action": "search_summary",
  "reason": "Explain why you want to search summaries",
  "limit" : "The number of results you want to retrieve",
  "query": "The search query to use",
  "urls": ["optional list of specific URLs to search in"] or null if not filtering by URLs
}}
```

For searching detailed content:
```json
{{
  "action": "search_full_content",
  "reason": "Explain why you want to search detailed content",
  "query": "The search query to use",
  "limit" : "The number of results you want to retrieve",
  "urls": ["list of specific URLs to search in"] or null if not filtering by URLs
}}
```

For insering supporting assertions:
```json
{{
  "action": "assert",
  "reason": "Explain why this information is useful to answer the question of the user",
  "answer": "The fact you have found which might help to answer the user question",
  "sources": [
    {{
      "resource_url": "URL or list of URLs of the source document/chunk",
      "resource_type": "Type of resource (summary or content)",
      "text": "Specific text from the source that supports your answer"
    }}
  ]
}}
```

For ending the process:
```json
{{
  "action": "answer",
  "reason": "Explain why you cannot answer or why no further research is needed",
  "answer": "Final explanation to the user" or null
}}
```
You have the option to search in the summaries and full context multiple times.
Whenever you feel that the assertions are is not sufficient to answer the user question you can search again
The information accessible by searching will not be updated while you are researching the report.
So if you search again in the same way twice you will get the same results. 
You can use your own query while searching but your query should be elaborative and relevant to the user question. This will reduce the number of searches you need to do.
Also instead of mentioning any generic things like today or now, you should use the date and time. 
It is not recommended to search for summaries repeatedly as it only gives a overview.
If you feel that you are not getting good results by searching summaries then you should search for full context instead.
Using same search method multiple times may not give you new results.

Pay attention to errors resulting from your previous actions.
Try to learn from them and not repeat them when you select further actions.


Include as much directly relevant information as possible from the data you obtain, including the assumptions. Try not to leave anything out.

Think carefully about whether you have enough information to answer the question accurately and completely.
Todays date is {today}.
"""


def build_select_action_prompt( state: RAGState) -> str:
        """Build prompt for action selection"""
        # Build conversation history context
        user_history_text = ""
        for item in state.user_conversation_history: 
            if item.get("role") == "human":
                user_history_text += f"[Human: {item.get('content')}]\n"
            else:
                user_history_text += f"{item.get('role', 'system').title()}: {item.get('content')}\n"
                
        # Build action history context
        history_text = ""
        for item in state.action_history[-5:]:  # Get last 5 items
            if item.get("role") == "system":
                history_text += f"[System: {item.get('content')}]\n"
            else:
                history_text += f"{item.get('role', 'unknown').title()}: {item.get('content')}\n"
        
        # Build summaries context
        summaries_text = ""
        for i, summary in enumerate(state.summaries):
            summaries_text += f"[{i+1}] URL: {summary['url']}\nSummary: {summary['content']}\n\n"
        
        # Build content chunks context
        content_text = ""
        for i, chunk in enumerate(state.content_chunks):
            content_text += f"[{i+1}] URL: {chunk['url']}\nContent: {chunk['content']}\n\n"
        
        # Build the prompt
        prompt = SELECT_ACTION_PROMPT.format(
            question=state.question,
            history=history_text,
            user_history=user_history_text,
            summaries=summaries_text if summaries_text else "No summaries available yet.",
            content_chunks=content_text if content_text else "No detailed content available yet.",
            research_steps=state.research_steps,
            max_research_steps=state.max_research_steps,
            today= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return prompt
    
async def select_action(state: Dict, ) -> Dict:
        """Decide what action to take next based on current state"""
        try:
            structured_state = state.get("structured", RAGState(question=state.get("question", "")))
            
            # Update step counter
            structured_state.research_steps += 1
            if structured_state.research_steps >= structured_state.max_research_steps:
                # Reached maximum steps, generate final answer
                state["destination"] = NodeLabels.GENERATE_ANSWER.value
                return state
            
            # Build prompt for action selection
            action_prompt =  build_select_action_prompt(structured_state)
            # Get LLM response
            llm_result =   LLM.invoke([HumanMessage(content=action_prompt)])
            # if  verbose:
            logger.info(f"LLM Response: {llm_result.content}")
            
            # Parse the response to get action
            response = SimpleJsonOutputParser().invoke(llm_result.content)
            action = Action(**response)
            
            # Update conversation history
            structured_state.action_history.append({
                "role": "assistant",
                "content": llm_result.content,
                "action": action.action
            })
            
            # Route based on action
            if action.action == ActionType.SEARCH_SUMMARY.value:
                state["destination"] = NodeLabels.SEARCH_SUMMARY.value
                state["query"] = action.query
                state["urls"] = action.urls
            elif action.action == ActionType.SEARCH_FULL_CONTENT.value:
                state["destination"] = NodeLabels.SEARCH_FULL_CONTENT.value
                state["query"] = action.query
                state["urls"] = action.urls
            elif action.action == ActionType.ANSWER.value:
                state["destination"] = NodeLabels.GENERATE_ANSWER.value
                state["answer"] = action.answer
                state["sources"] = action.sources
            
            # Update state
            state["structured"] = structured_state
            return state
            
        except (OutputParserException, ValidationError) as e:
            # Handle parsing errors
            error = Error(
                exception=str(e),
                type=str(type(e).__name__),
                detail="Failed to parse LLM response",
                data={"response": llm_result.content if 'llm_result' in locals() else None}
            )
            
            if "structured_state" in locals():
                structured_state.errors.append(error)
                state["structured"] = structured_state
            
            # Default to searching summaries on error
            state["destination"] = NodeLabels.SEARCH_SUMMARY.value
            state["query"] = state.get("question", "")
            state["urls"] = None
            
            logger.error(f"Error in select_action: {e}")
            return state

async def search_summary( state: Dict) -> Dict:
        """Search in summary collection based on query"""
        try:
            structured_state = state.get("structured")
            query = state.get("query")
            urls = state.get("urls")
            
            # Call the search tool
            search_request = SearchRequest(query=query, limit=5, urls=urls)
            # tool_action = ToolInvocation(tool="search_summary", tool_input=search_request.dict())
            # search_results = await tool_executor.ainvoke(tool_action)
            
            # Initialize tools
            search_summary_tool = SearchSummaryTool("search_summary")
            tool_node = ToolNode([search_summary_tool])
            tool_call = ToolCall(
                name="search_summary",
                args=search_request.dict(),
                id="search_summary"+"___"+str(uuid.uuid4()),
            )
            ai_message = AIMessage(
                content="",
                tool_calls=[tool_call]
            )
            tool_output =  await tool_node.ainvoke({"messages":[ai_message]})
            msg = HumanMessage(content=tool_output['messages'][0].content)
            search_results = SimpleJsonOutputParser().invoke(msg)
            # Update state with search results
            structured_state.summaries = search_results
            
            # Convert results to resources and add to valid sources
            for result in search_results:
                resource = Resource(
                    resource_url=result['url'],
                    resource_type=ResourceType.SUMMARY.value,
                    text=result['content'],
                    metadata=result['metadata']
                )
                structured_state.valid_sources.append(resource)
            
            # Add search results to conversation history
            structured_state.action_history.append({
                "role": "system",
                "content": f"Searched summaries for: {query}",
                "results_count": len(search_results)
            })
            
            state["structured"] = structured_state
            return state
            
        except Exception as e:
            # Handle errors
            error = Error(
                exception=str(e),
                type=str(type(e).__name__),
                detail="Error searching summaries",
                data={"query": state.get("query")}
            )
            
            if structured_state:
                structured_state.errors.append(error)
                state["structured"] = structured_state
            
            logger.error(f"Error in search_summary: {e}")
            return state

async def search_full_content( state: Dict) -> Dict:
        """Search in full content collection based on query"""
        try:
            structured_state = state.get("structured")
            query = state.get("query")
            urls = state.get("urls")
            
            # Call the search tool
            search_request = SearchRequest(query=query, limit=5, urls=urls)
            # tool_action = ToolInvocation(tool="search_full_content", tool_input=search_request.dict())
            # search_results = await self.tool_executor.ainvoke(tool_action)
            search_full_content_tool = SearchFullContentTool("search_full_content")
            tool_node = ToolNode([search_full_content_tool])
            tool_call = ToolCall(
                name="search_full_content",
                args=search_request.dict(),
                id="search_full_content"+"___"+str(uuid.uuid4()),
            )
            ai_message = AIMessage(
                content="",
                tool_calls=[tool_call]
            )
            tool_output =  await tool_node.ainvoke({"messages":[ai_message]})
            msg = HumanMessage(content=tool_output['messages'][0].content)
            search_results = SimpleJsonOutputParser().invoke(msg)
            
            # Update state with search results
            structured_state.content_chunks = search_results
            
            # Convert results to resources and add to valid sources
            for result in search_results:
                resource = Resource(
                    resource_url=f"{result['url']}_{result['metadata'].get('chunk_id', '')}",
                    resource_type=ResourceType.CONTENT.value,
                    text=result['content'],
                    metadata=result['metadata']
                )
                structured_state.valid_sources.append(resource)
            
            # Add search results to conversation history
            structured_state.action_history.append({
                "role": "system",
                "content": f"Searched full content for: {query}",
                "results_count": len(search_results)
            })
            
            state["structured"] = structured_state
            return state
            
        except Exception as e:
            # Handle errors
            error = Error(
                exception=str(e),
                type=str(type(e).__name__),
                detail="Error searching full content",
                data={"query": state.get("query")}
            )
            
            if structured_state:
                structured_state.errors.append(error)
                state["structured"] = structured_state
            
            logger.error(f"Error in search_full_content: {e}")
            return state


GENERATE_ANSWER_PROMPT = """You are an intelligent research assistant that helps answer questions by analyzing relevant information.

Your task is to generate a comprehensive answer to the user's question based on the retrieved information.

QUESTION: {question}

AVAILABLE SOURCES:
{sources}

CHAT HISTORY:
{user_history}

Generate a well-structured, comprehensive answer to the question based on the available sources. 
Include specific details and information from the sources to support your answer.
Make sure to cite the specific sources used in your answer.
Also make sure you answer from only the sources which have only a valid resource_url.  
Don't consider any source which don't have a resource_url. 


Respond with a JSON object:
```json
{{
  "answer": "Your comprehensive answer to the user's question",
  "sources": [
    {{
      "resource_url": "URL or list of URLs of the source document/chunk",
      "resource_type": "Type of resource (summary or content)",
      "text": "Specific text from the source that supports your answer"
    }}
  ]
}}
```
Make sure you attach the exact url in the "resource_url" field of the "sources". Don't mention generic terms like summary_1 , context_1 etc.
Don't attach urls or make any citation inside your answer. Answer is meant to be read by a human. 
You have to attach the URLs in the "resource_url" field only inside "sources" key.
You answer should be the collected from the "text" available in the "sources" , Not the URLs or types.
Keep that in mind you have to answer the user's question based on the sources you have. 
So your approach should be like an QA assistant. 

As user may ask sub sequent questions based on the final answer generated by you.
You should look into the chat history to unerstand the user need better.

If you can't confidently answer the question with the available information, say so clearly in your answer.
But you find that the sources have relevant information about the user question but does not directly or 
completely answers the user question then you may answer on the partial or relevant information available with you
but make sure to mention this thing to the user
"""


def build_answer_prompt( state: RAGState) -> str:
        """Build prompt for answer generation"""
        # Build conversation history context
        user_history_text = ""
        for item in state.user_conversation_history: 
            if item.get("role") == "human":
                user_history_text += f"[Human: {item.get('content')}]\n"
            else:
                user_history_text += f"{item.get('role', 'system').title()}: {item.get('content')}\n"
                
        # Build sources context
        sources_text = ""
        for i, source in enumerate(state.valid_sources):
            if source.resource_type == ResourceType.SUMMARY.value:
                sources_text += f"[Summary {i+1}] ID: {source.resource_url}\nContent: {source.text}\n\n"
            elif source.resource_type == ResourceType.CONTENT.value:
                sources_text += f"[Content {i+1}] ID: {source.resource_url}\nContent: {source.text}\n\n"
        
        # Build the prompt
        prompt = GENERATE_ANSWER_PROMPT.format(
            question=state.question,
            sources=sources_text,
            user_history=user_history_text
        )
        
        return prompt

async def generate_answer(state: Dict) -> Dict:
    """Generate a final answer based on all gathered information"""
    try:
        structured_state = state.get("structured")

        # Build prompt for answer generation
        answer_prompt = build_answer_prompt(structured_state)

        # Get LLM response
        llm_result = await LLM.ainvoke([HumanMessage(content=answer_prompt)])
        logger.info(f"generate_answer llm result: {llm_result.content}")

        # Parse the response
        try:
            response = SimpleJsonOutputParser().invoke(llm_result.content)
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            logger.info(f"Answer parsed successfully by generate_answer: {answer}")
            logger.info(f"sources generated by generate_answer: {sources}")

        except (OutputParserException, ValidationError) as e:
            # If parsing fails, use the raw response
            answer = llm_result.content
            sources = []
            logger.info(f"parsing error by generate_answer: {e}")


        # Add final answer to conversation history
        structured_state.action_history.append({
            "role": "assistant",
            "content": answer,
            "is_final_answer": True
        })

        # Prepare final result
        state["final_answer"] = answer
        state["sources"] = sources
        state["structured"] = structured_state
        state["destination"] = NodeLabels.REPORT_RESULT.value

        return state

    except Exception as e:
        # Handle errors
        error = Error(
            exception=str(e),
            type=str(type(e).__name__),
            detail="Error generating answer",
            data={}
        )
        
        if structured_state:
            structured_state.errors.append(error)
            state["structured"] = structured_state
        logger.error(f"Error in generate_answer: {e}")
        # Return a fallback answer
        state["final_answer"] = "I'm sorry, I encountered an error while generating your answer."
        state["sources"] = []
        state["destination"] = NodeLabels.REPORT_RESULT.value
        
        logger.error(f"Error in generate_answer: {e}")
        return state


def report_result(state: Dict) -> Dict:
        """Format and return the final result"""
        final_answer = state.get("final_answer", "")
        sources = state.get("sources", [])
        structured_state = state.get("structured")
        logger.info(f"Final Answer: {final_answer}")
        logger.info(f"Sources Found: {sources}")

        result = {
            "answer": final_answer,
            "sources": sources,
            "action_history": structured_state.action_history if structured_state else [],
            "errors": [e.dict() for e in structured_state.errors] if structured_state else []
        }
        
        return result

def route(state: Dict) -> str:
        """Route to the next node based on the chosen action"""
        logger.info(f"Routing to: {state['destination']}")
        return state["destination"]


class NodeLabels(str, Enum):
    SELECT_ACTION = "select_action"
    SEARCH_SUMMARY = "search_summary"
    SEARCH_FULL_CONTENT = "search_full_content" 
    GENERATE_ANSWER = "generate_answer"
    REPORT_RESULT = "report_result"

workflow = StateGraph(Dict)

workflow.add_node(NodeLabels.SELECT_ACTION.value,  select_action)
workflow.add_node(NodeLabels.SEARCH_SUMMARY.value,  search_summary)
workflow.add_node(NodeLabels.SEARCH_FULL_CONTENT.value,  search_full_content)
workflow.add_node(NodeLabels.GENERATE_ANSWER.value,  generate_answer)
workflow.add_node(NodeLabels.REPORT_RESULT.value,  report_result)

# Set entry and finish points
workflow.set_entry_point(NodeLabels.SELECT_ACTION.value)
workflow.set_finish_point(NodeLabels.REPORT_RESULT.value)

# Add conditional edges
workflow.add_conditional_edges(
    NodeLabels.SELECT_ACTION.value,
     route,
    {
        NodeLabels.SEARCH_SUMMARY.value: NodeLabels.SEARCH_SUMMARY.value,
        NodeLabels.SEARCH_FULL_CONTENT.value: NodeLabels.SEARCH_FULL_CONTENT.value,
        NodeLabels.GENERATE_ANSWER.value: NodeLabels.GENERATE_ANSWER.value,
        NodeLabels.REPORT_RESULT.value: NodeLabels.REPORT_RESULT.value
    }
)

# Add direct edges
workflow.add_edge(NodeLabels.SEARCH_SUMMARY.value, NodeLabels.SELECT_ACTION.value)
workflow.add_edge(NodeLabels.SEARCH_FULL_CONTENT.value, NodeLabels.SELECT_ACTION.value)
workflow.add_edge(NodeLabels.GENERATE_ANSWER.value, NodeLabels.REPORT_RESULT.value)

app = workflow.compile()

