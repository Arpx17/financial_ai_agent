import streamlit as st
import asyncio
import json
import logging
import re
import time
import torch
from typing import List, Tuple, Dict
import uuid
from datetime import datetime

# Import NER model functions
from transformers import PreTrainedTokenizerBase
from unsloth import FastLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NER Financial Document Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .ner-highlight {
        padding: 2px 4px;
        border-radius: 3px;
        margin: 0 2px;
    }
        .ner-ACCOUNT_PIN { background-color: #ffadad; }
    .ner-API_KEY { background-color: #ffd6a5; }
    .ner-BANK_ROUTING_NUMBER { background-color: #fdffb6; }
    .ner-BBAN { background-color: #caffbf; }
    .ner-COMPANY { background-color: #9bf6ff; }
    .ner-CREDIT_CARD_NUMBER { background-color: #a0c4ff; }
    .ner-CREDIT_CARD_SECURITY_CODE { background-color: #bdb2ff; }
    .ner-CUSTOMER_ID { background-color: #ffc6ff; }
    .ner-DATE { background-color: #b8e0d2; }
    .ner-DATE_OF_BIRTH { background-color: #ffb6c1; }
    .ner-DRIVER_LICENSE_NUMBER { background-color: #f1cbff; }
    .ner-EMAIL { background-color: #d8bfd8; }
    .ner-EMPLOYEE_ID { background-color: #f5e0b7; }
    .ner-FIRST_NAME { background-color: #b0e57c; }
    .ner-IBAN { background-color: #ffdac1; }
    .ner-IPV4 { background-color: #bae1ff; }
    .ner-IPV6 { background-color: #ffffd1; }
    .ner-LAST_NAME { background-color: #f3b0c3; }
    .ner-LOCAL_LATLNG { background-color: #d0f4de; }
    .ner-NAME { background-color: #e2f0cb; }
    .ner-PASSPORT_NUMBER { background-color: #f7c59f; }
    .ner-PASSWORD { background-color: #c1c8e4; }
    .ner-SSN { background-color: #e4c1f9; }
    .ner-STREET_ADDRESS { background-color: #ffcbf2; }
    .ner-SWIFT_BIC_CODE { background-color: #cbf3f0; }
    .ner-TIME { background-color: #ede0d4; }
    .ner-USER_NAME { background-color: #f6bd60; }
    .ner-legend {
        display: flex;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .ner-legend-item {
        display: flex;
        align-items: center;
        margin-right: 15px;
        margin-bottom: 8px;
    }
    .ner-legend-color {
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border-radius: 3px;
    }
    .ner-stats {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .processing-time {
        font-style: italic;
        color: #666;
        margin-top: 10px;
    }
    .search-result {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "ner_entity_dict" not in st.session_state:
        st.session_state.ner_entity_dict = {
    "ACCOUNT_PIN": "Personal Identification Number for accounts",
    "API_KEY": "Authentication token for accessing APIs",
    "BANK_ROUTING_NUMBER": "Bank’s routing number used in financial transactions",
    "BBAN": "Basic Bank Account Number used in international banking",
    "COMPANY": "Name of a business or organization",
    "CREDIT_CARD_NUMBER": "Number on a credit card used for payments",
    "CREDIT_CARD_SECURITY_CODE": "CVV or security code on a credit card",
    "CUSTOMER_ID": "Unique identifier assigned to a customer",
    "DATE": "Calendar date (e.g. 2025-04-10)",
    "DATE_OF_BIRTH": "Person’s date of birth",
    "DRIVER_LICENSE_NUMBER": "License number assigned to a driver",
    "EMAIL": "Email address",
    "EMPLOYEE_ID": "Identifier assigned to an employee",
    "FIRST_NAME": "Given name of a person",
    "IBAN": "International Bank Account Number used in global banking",
    "IPV4": "IPv4 format IP address (e.g. 192.168.1.1)",
    "IPV6": "IPv6 format IP address",
    "LAST_NAME": "Surname or family name of a person",
    "LOCAL_LATLNG": "Geolocation data in latitude and longitude format",
    "NAME": "Full name of a person",
    "PASSPORT_NUMBER": "Number identifying a person’s passport",
    "PASSWORD": "Confidential login credential",
    "SSN": "Social Security Number (often US)",
    "STREET_ADDRESS": "Street-level residential or business address",
    "SWIFT_BIC_CODE": "Bank Identifier Code used for international wires",
    "TIME": "Time of day (e.g. 14:00, 2 PM)",
    "USER_NAME": "Username used for login or identification"}
    if "ner_model" not in st.session_state:
        st.session_state.ner_model = None
    if "ner_tokenizer" not in st.session_state:
        st.session_state.ner_tokenizer = None
    if "ner_model_loaded" not in st.session_state:
        st.session_state.ner_model_loaded = False
    if "search_tools" not in st.session_state:
        st.session_state.search_tools = {}
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "selected_text" not in st.session_state:
        st.session_state.selected_text = ""
    if "ner_results" not in st.session_state:
        st.session_state.ner_results = []
    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0

# Call initialization
init_session_state()

# Load NER model resources
@st.cache_resource
def load_ner_model():
    """Load the NER model - cached to prevent reloading"""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Arpx22/llama-3.2-1B-Finetuned-ner-finance",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
            gpu_memory_utilization=0.5
        )
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.2",
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            map_eos_token=True
        )
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading NER model: {str(e)}")
        return None, None

# Load search tools if not loaded
@st.cache_resource
def load_search_tools():
    """Load search tools - cached to prevent reloading"""
    try:
        from financial_ai_agent.agent.search_tools import SearchFullContentTool
        simple_search_tool = SearchFullContentTool("simple_search")
        return {"simple": simple_search_tool}
    except Exception as e:
        logger.error(f"Error loading search tools: {str(e)}")
        return {}

# Helper functions for NER
def testing_prompt_xml(
    description_dict: dict, 
    input_str: str,
    tokenizer: PreTrainedTokenizerBase,
):
    """Create a prompt for NER detection using the specified model."""
    
    usr_msg1 = f"""You are given a user utterance that may contain Personal Identifiable Information (PII). 
        You are also given a list of entity types representing Personal Identifiable Information (PII). 
        Your task is to detect and identify all instances of the supplied PII entity types in the user utterance. 
        The output must have the same content as the input. Only the tokens that match the PII entities in the 
        list should be enclosed within XML tags. The XML tag comes from the PII entities described in the list below. 
        For example, a given name of a person should be enclosed within <FIRST_NAME></FIRST_NAME> tags.
        Ensure that all entities are identified. Do not perform false identifications.
        \n\nList Of Entities\n{json.dumps(description_dict,indent=2)}
        \n\n
        Are the instructions clear to you?"""
    
    asst_msg1 = """Yes, the instructions are clear. I will identify and enclose within the corresponding XML tags, 
        all instances of the specified PII entity types in the user utterance. For example, 
        <FIRST_NAME><Given name of a person></FIRST_NAME>, <PASSPORT_NUMBER><Number identifying a person's passport></PASSPORT_NUMBER>, etc. 
        leaving the rest of the user utterance unchanged."""
    
    usr_msg2 = "My name is John Doe, and I can be contacted at meet_me_john@gmail.com"
    
    asst_msg2 = "My name is <FIRST_NAME>John</FIRST_NAME> <LAST_NAME>Doe</LAST_NAME>, and I can be contacted at <EMAIL>meet_me_john@gmail.com</EMAIL>"
    
    usr_msg3 = "Give a brief explanation of why your answer is correct."

    asst_msg3 = """My answer is correct because I identified the specified PII entity types in the user utterance and enclosed them within the corresponding XML tags.

                    - "John" is a given name, so it was enclosed within <FIRST_NAME> tags.
                    - "Doe" is a surname, so it was enclosed within <LAST_NAME> tags.
                    - "meet_me_john@gmail.com" is an email address, so it was enclosed within <EMAIL> tags."""
    
    usr_msg4 = """Great! I am now going to give you another user utterance. Please detect PII entities in it according to the previous instructions. Do not include an explanation in your answer."""
    
    asst_msg4 = "Sure! Please give me the user utterance."

    messages = [
        {"role": "user", "content": usr_msg1},
        {"role": "assistant", "content": asst_msg1},
        {"role": "user", "content": usr_msg2},
        {"role": "assistant", "content": asst_msg2},
        {"role": "user", "content": usr_msg3},
        {"role": "assistant", "content": asst_msg3},
        {"role": "user", "content": usr_msg4},
        {"role": "assistant", "content": asst_msg4},
        {"role": "user", "content": input_str},
    ]
    
    encoded_input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    if torch.cuda.is_available():
        encoded_input_ids = encoded_input_ids.to("cuda")
    
    return encoded_input_ids

def extract_outermost_entities(text: str, valid_entity_types: List[str]) -> List[Tuple[str, str]]:
    """Extract entity mentions from tagged text."""
    entities = []
    
    for entity in valid_entity_types:
        # Match the outermost <ENTITY> ... </ENTITY> including malformed variants
        pattern = fr"""
            (<+{entity}>+)            # Opening tag (malformed tolerated)
            (.*?)                     # Non-greedy capture of content
            (</+{entity}>+)           # Closing tag (malformed tolerated)
        """
        # Use re.DOTALL to match across lines and VERBOSE for formatting
        matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.VERBOSE))
        
        # To avoid nested matches, we skip overlaps
        used_spans = set()
        for match in matches:
            span = match.span()
            if any(start <= span[0] < end or start < span[1] <= end for start, end in used_spans):
                continue  # Skip if overlaps with already accepted match
            
            inner_text = match.group(2).strip()

            # Remove nested tags inside entity text
            inner_text = re.sub(r"<[^>]+>", "", inner_text).strip()
            cleaned_text = re.sub(r'{}'.format(entity), '', inner_text)
            cleaned_text = re.sub(r'[<>/\\\n\t]', '', cleaned_text)
            cleaned_text = cleaned_text.strip()
        
            # Deduplicate repeating sequences of words using regex
            pattern = r'\b(\w+(?:\s+\w+)*)\b(?:\s+\1\b)+'
            deduplicated_text = re.sub(pattern, r'\1', cleaned_text)
            
            entities.append((entity, deduplicated_text))
            used_spans.add(span)
    
    # Filter out existing entities and empty values
    entities = list(set(entities))
    existing_ent = [('PASSPORT_NUMBER', ''),('LAST_NAME', 'Doe'),('FIRST_NAME', 'John'),('EMAIL', 'meet_me_john@gmail.com'),('FIRST_NAME', '')]
    entities = [ent for ent in entities if ent not in existing_ent and ent[1].strip()]
    
    return list(set(entities))


def tag_or_highlight_entities(
    text: str,
    entities: List[Tuple[str, str]],
    mode: str = "tag"  # "tag" or "highlight"
) -> str:
    # Define colors for each entity type
    entity_colors = {
        "ACCOUNT_PIN": "#ffadad",
        "API_KEY": "#ffd6a5",
        "BANK_ROUTING_NUMBER": "#fdffb6",
        "BBAN": "#caffbf",
        "COMPANY": "#9bf6ff",
        "CREDIT_CARD_NUMBER": "#a0c4ff",
        "CREDIT_CARD_SECURITY_CODE": "#bdb2ff",
        "CUSTOMER_ID": "#ffc6ff",
        "DATE": "#b8e0d2",
        "DATE_OF_BIRTH": "#ffb6c1",
        "DRIVER_LICENSE_NUMBER": "#f1cbff",
        "EMAIL": "#d8bfd8",
        "EMPLOYEE_ID": "#f5e0b7",
        "FIRST_NAME": "#b0e57c",
        "IBAN": "#ffdac1",
        "IPV4": "#bae1ff",
        "IPV6": "#ffffd1",
        "LAST_NAME": "#f3b0c3",
        "LOCAL_LATLNG": "#d0f4de",
        "NAME": "#e2f0cb",
        "PASSPORT_NUMBER": "#f7c59f",
        "PASSWORD": "#c1c8e4",
        "SSN": "#e4c1f9",
        "STREET_ADDRESS": "#ffcbf2",
        "SWIFT_BIC_CODE": "#cbf3f0",
        "TIME": "#ede0d4",
        "USER_NAME": "#f6bd60"
    }

    # Sort entities to tag longer ones first to avoid substring overlap
    entities = sorted(entities, key=lambda x: -len(x[1]))

    # Track used spans to avoid overlapping replacements
    used_spans = []
    replacements = []

    for entity_type, entity_value in entities:
        entity_value_escaped = re.escape(entity_value)
        pattern = re.compile(entity_value_escaped)

        for match in pattern.finditer(text):
            start, end = match.span()
            if any(s < end and start < e for s, e in used_spans):
                continue  # Skip overlapping match

            used_spans.append((start, end))

            if mode == "highlight":
                color = entity_colors.get(entity_type.upper(), "#e0e0e0")
                replacement = (
                    f'<span class="ner-highlight ner-{entity_type.lower()}" '
                    f'style="background-color:{color}; padding:2px 4px; border-radius:4px;">'
                    f'{match.group(0)}</span>'
                )
            else:
                replacement = f"<{entity_type}>{match.group(0)}</{entity_type}>"

            replacements.append((start, end, replacement))

    # Sort replacements by start index descending so we don't mess up indices
    replacements.sort(reverse=True)

    for start, end, replacement in replacements:
        text = text[:start] + replacement + text[end:]

    return text


# Functions for search and NER processing
async def perform_search(query: str):
    """Search for documents related to the query."""
    if not st.session_state.search_tools:
        st.session_state.search_tools = load_search_tools()
    
    if not st.session_state.search_tools:
        return {
            "error": "Search tools could not be loaded.",
            "results": []
        }
    
    try:
        # Create search request
        from financial_ai_agent.agent.types import SearchRequest
        
        search_request = SearchRequest(query=query, limit=3)
        
        # Get search tool
        search_tool = st.session_state.search_tools["simple"]
        
        # Direct invocation of search tool
        search_results = await search_tool._arun(
            query=query, 
            limit=3
        )
        
        return {
            "error": None,
            "results": search_results
        }
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return {
            "error": f"Error searching: {str(e)}",
            "results": []
        }

def process_ner(text: str):
    """Process named entity recognition on the given text."""
    if not st.session_state.ner_model_loaded:
        with st.spinner("Loading NER model..."):
            st.session_state.ner_model, st.session_state.ner_tokenizer = load_ner_model()
            if st.session_state.ner_model and st.session_state.ner_tokenizer:
                st.session_state.ner_model_loaded = True
            else:
                return {
                    "error": "Could not load NER model",
                    "tagged_text": "",
                    "entities": []
                }
    
    start_time = time.time()
    
    try:
        # Generate encoded input
        encoded_inputs = testing_prompt_xml(
            st.session_state.ner_entity_dict,
            text,
            st.session_state.ner_tokenizer
        )
        
        # Generate output with model
        with st.spinner("Identifying named entities..."):
            outputs = st.session_state.ner_model.generate(
                encoded_inputs, 
                max_new_tokens=4096,  # Adjusted for longer texts
                use_cache=True
            )
        
        # Decode output
        output_text = st.session_state.ner_tokenizer.batch_decode(outputs)[0]
        
        # Extract entities
        entities = extract_outermost_entities(
            output_text, 
            list(st.session_state.ner_entity_dict.keys())
        )
        
        tagged_text = tag_or_highlight_entities(text,entities,"tag")

        processing_time = time.time() - start_time
        
        return {
            "error": None,
            "tagged_text": tagged_text,
            "entities": entities,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error in NER processing: {e}")
        return {
            "error": f"Error processing NER: {str(e)}",
            "tagged_text": "",
            "entities": [],
            "processing_time": time.time() - start_time
        }
# def mock_process_ner(text: str):
#     txt =  """<COMPANY>Indian Overseas Bank</COMPANY> reduces Repo Linked Lending Rate by 25 basis points
#     Indian Overseas Bank has lowered its Repo Linked Lending Rate by 25 basis points, effective immediately. This decision follows the Reserve B...

#     Delhi: Stampede-like situation reported at Indira Gandhi International Airport as 50 flights get delayed due to dust storm
#     A severe dust storm paralyzed Delhi and the NCR, causing widespread flight disruptions at Indira Gandhi International Airport. Over 50 domes...

#     India's auto component industry to hit USD 145 billion by <DATE>2030</DATE>, exports to triple: <COMPANY>NITI Aayog</COMPANY>
#     NITI Aayog envisions India's automotive component production soaring to USD 145 billion by 2030, with exports tripling to USD 60 billion. A ...

#     Banks scouting buyers for Sahara Star Hotel loans
#     Lenders led by <COMPANY>Union Bank of India</COMPANY> are set to auction the Rs 700 crore debt owed by Sahara Star Hotel after the insolvency plea was rejected...

#     Global media titans to converge in Mumbai for first-ever WAVES summit in <DATE>May</DATE>
#     Mumbai is gearing up to host the inaugural World Audio Visual & Entertainment Summit (WAVES) from <DATE>May 1-4</DATE>, backed by the Indian government. ...

#     India is the place to be: Radisson Top Exec
#     <COMPANY>Radisson Hotel Group</COMPANY> is bullish on India's hotel development potential, aiming to reach 500 hotels by <DATE>the end of the decade</DATE> by signing 50 """
#     return {
#             "error": None,
#             "tagged_text": txt,
#             "entities": [('COMPANY', 'Radisson Hotel Group'),
#                             ('COMPANY', 'Indian Over'),
#                             ('DATE', 'May'),
#                             ('DATE', 'May 1-4'),
#                             ('DATE', 'the end of the decade'),
#                             ('COMPANY', 'Union Bank of India'),
#                             ('COMPANY', 'Indian Overseas Bank'),
#                             ('DATE', '2030'),
#                             ('COMPANY', 'NITI Aayog')],
#             "processing_time": 100
#         }

# UI Components
def render_sidebar():
    """Render sidebar with controls and settings."""
    with st.sidebar:
        st.header("NER Settings")
        
        st.subheader("Model Information")
        st.info("Using LLaMA 3.2 1B model fine-tuned for financial NER")
        
        st.subheader("Entity Types")
        for entity, description in st.session_state.ner_entity_dict.items():
            st.markdown(f"**{entity}**: {description}")
        
        st.subheader("Performance Tips")
        st.markdown("""
        - Processing longer texts takes more time
        - Using specific search queries helps find relevant documents
        - For best results, select concise text portions for analysis
        """)

def render_entity_legend():
    """Render color-coded legend for entity types."""
    st.markdown("### Entity Types Legend")
    
    legend_html = '\n<div class="ner-legend">'
    for entity in st.session_state.ner_entity_dict.keys():
        legend_html +=  f"""\n\t<div class="ner-legend-item">\n\t\t<div class="ner-legend-color ner-{entity}"></div>\n\t\t<div>{entity}</div>\n\t</div>"""
    legend_html += '\n</div>'
    
    st.markdown(legend_html, unsafe_allow_html=True)

def render_entity_stats(entities):
    """Render statistics about detected entities."""
    entity_counts = {}
    for entity_type, _ in entities:
        if entity_type not in entity_counts:
            entity_counts[entity_type] = 0
        entity_counts[entity_type] += 1
    
    if entity_counts:
        st.markdown("### Entity Statistics")
        stats_html = '<div class="ner-stats">'
        stats_html += f"<p>Total entities detected: {len(entities)}</p>"
        stats_html += "<ul>"
        for entity_type, count in entity_counts.items():
            stats_html += f"<li><strong>{entity_type}</strong>: {count}</li>"
        stats_html += "</ul></div>"
        
        st.markdown(stats_html, unsafe_allow_html=True)
    else:
        st.info("No entities detected in the analyzed text.")

# Main app layout
st.title("Financial Document NER Analyzer")
st.markdown("""
This tool allows you to search for financial documents and analyze them for Named Entity Recognition (NER).
First, search for relevant documents, then select text to analyze for personally identifiable information (PII).
""")

# Render sidebar
render_sidebar()

# Search section
st.header("1. Search for Financial Documents")
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Enter your search query:", 
                                placeholder="Example: quarterly earnings report for Apple")
with col2:
    search_button = st.button("Search Documents", use_container_width=True)

# Process search
if search_button and search_query:
    with st.spinner("Searching for documents..."):
        search_response = asyncio.run(perform_search(search_query))
        
        if search_response["error"]:
            st.error(search_response["error"])
        else:
            st.session_state.search_results = search_response["results"]
            st.success(f"Found {len(st.session_state.search_results)} documents")

# Display search results
if st.session_state.search_results:
    st.header("2. Select Document to Analyze")
    
    for idx, result in enumerate(st.session_state.search_results):
        with st.expander(f"Document {idx+1}: {result.get('title', 'Untitled Document')[:50]}...", expanded=idx==0):
            st.markdown(f"<div class='search-result'>", unsafe_allow_html=True)
            st.markdown(f"**Source:** [{result.get('url', 'Unknown Source')}]({result.get('url', '#')})")
            
            # Show document content with select button
            st.markdown("**Content:**")
            st.text_area("Document content", value=result.get('content', 'No content available'), 
                        height=150, key=f"doc_content_{idx}")
            
            if st.button("Select for NER Analysis", key=f"select_doc_{idx}"):
                st.session_state.selected_text = result.get('content', '')
            
            st.markdown("</div>", unsafe_allow_html=True)

# Text selection and NER analysis
if st.session_state.selected_text:
    st.header("3. Named Entity Recognition Analysis")
    
    # Option to edit selected text
    edited_text = st.text_area("Edit text before analysis (if needed)", 
                             value=st.session_state.selected_text, 
                             height=200)
    
    analyze_button = st.button("Run NER Analysis", type="primary", use_container_width=True)
    
    if analyze_button:
        # Run NER analysis
        ner_response = process_ner(edited_text)
        
        if ner_response["error"]:
            st.error(ner_response["error"])
        else:
            st.session_state.ner_results = ner_response["entities"]
            st.session_state.processing_time = ner_response["processing_time"]
            
            st.success(f"Analysis complete! Found {len(st.session_state.ner_results)} entities.")
            st.markdown(f"<p class='processing-time'>Processing time: {st.session_state.processing_time:.2f} seconds</p>", 
                      unsafe_allow_html=True)
            
            # Display colored legend
            render_entity_legend()
            
            # Display entity statistics
            render_entity_stats(st.session_state.ner_results)
            
            # Display highlighted text
            st.subheader("Text with Highlighted Entities")
            highlighted_html = tag_or_highlight_entities(edited_text, st.session_state.ner_results,'highlight')
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # Display raw tagged output for debugging
            with st.expander("View Raw Tagged Output", expanded=False):
                st.text(ner_response["tagged_text"])