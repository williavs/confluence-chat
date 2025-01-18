import streamlit as st
import asyncio
import base64
import logging
import os
import requests
import time
from llm_logic import LLMHandler, clean_html_content

# Must be the first Streamlit command
st.set_page_config(
    page_title="Confluence Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown(
        """
        <style>
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .rainbow-text {
            background: linear-gradient(
                45deg,
                #0070D4,#ffffff, #FF9F82, #00A46B, #F3B715, #FB7F59, #ffffff
            );
            background-size: 400% 400%;
            color: transparent;
            -webkit-background-clip: text;
            background-clip: text;
            animation: gradient 5s ease infinite;
            font-weight: heavy;
            font-size: 18px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

# Call this function at the beginning of your app
inject_custom_css()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for API keys if not present
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "confluence_api_token" not in st.session_state:
    st.session_state.confluence_api_token = ""
if "confluence_email" not in st.session_state:
    st.session_state.confluence_email = ""

# Sidebar API key management
with st.sidebar:
    st.divider()
    st.markdown("### 🔑 API Configuration")
    
    # OpenAI API key input
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys",
        key="openai_key_input"
    )
    
    # Confluence API configuration
    confluence_token = st.text_input(
        "Confluence API Token",
        type="password",
        value=st.session_state.confluence_api_token,
        help="Enter your Confluence API token. Generate one in your Confluence settings.",
        key="confluence_token_input"
    )
    
    confluence_email = st.text_input(
        "Confluence Email",
        value=st.session_state.confluence_email,
        help="Enter your Confluence account email",
        key="confluence_email_input"
    )
    
    if st.button("Save API Configuration", type="primary"):
        if openai_key and confluence_token and confluence_email:
            st.session_state.openai_api_key = openai_key
            st.session_state.confluence_api_token = confluence_token
            st.session_state.confluence_email = confluence_email
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("✅ API configuration saved!")
            st.rerun()
        else:
            st.error("Please fill in all API configuration fields")

# Check for required API keys
if not st.session_state.openai_api_key or not st.session_state.confluence_api_token or not st.session_state.confluence_email:
    st.error("⚠️ Please configure your API keys in the sidebar")
    st.stop()

# Set OpenAI API key for LangChain
os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

# Confluence API configuration
BASE_URL = "https://justworks.atlassian.net/wiki/api/v2"
EMAIL = st.session_state.confluence_email
API_TOKEN = st.session_state.confluence_api_token
auth_header = {"Authorization": "Basic " + base64.b64encode(f"{EMAIL}:{API_TOKEN}".encode()).decode()}

# Define the specific pages
CONFLUENCE_SPACE = "~524722389"  # Personal space ID
CONFLUENCE_BASE_URL = "https://justworks.atlassian.net/wiki/spaces"

# Documentation structure - This will be populated from schema_config
PAGES = {}

def api_get(url, params=None):
    try:
        response = requests.get(url, headers=auth_header, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        return None

def get_page_content(page_id: str) -> dict:
    """Retrieve the content of a Confluence page."""
    params = {"body-format": "storage"}
    return api_get(f"{BASE_URL}/pages/{page_id}", params=params)

def get_confluence_url(page):
    """Generate Confluence URL based on space and page ID."""
    return f"{CONFLUENCE_BASE_URL}/{page['space']}/pages/{page['page_id']}"

@st.cache_resource
def get_documentation_content():
    """Fetch and combine content from all specified pages."""
    try:
        # Load schema from session state if available
        if 'schema_config' in st.session_state:
            PAGES.update(st.session_state.schema_config)
        
        combined_content = []
        for category, pages in PAGES.items():
            combined_content.append(f"\n=== {category} ===\n")
            for page in pages:
                content = get_page_content(page["page_id"])
                if content and 'body' in content:
                    html_content = content['body']['storage']['value']
                    clean_content = clean_html_content(html_content)
                    combined_content.append(f"--- {page['title']} ---\n{clean_content}\n")
        
        return "\n".join(combined_content)
    except Exception as e:
        logger.error(f"Error fetching documentation: {str(e)}")
        raise

# Initialize LLM handler
llm_handler = LLMHandler(get_documentation_content)

async def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton button {
            width: 100%;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stMarkdown {
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with documentation sections
    with st.sidebar:
        st.title("Documentation Assistant")
        
        if st.button("🔄 New Chat", key="sidebar_new_chat", use_container_width=True):
            st.session_state.messages = []
            llm_handler.clear_history()
            st.rerun()
        
        st.sidebar.divider()
        
        st.markdown("### Documentation Categories")
        
        # Display documentation categories and pages
        if 'schema_config' in st.session_state:
            for category, pages in st.session_state.schema_config.items():
                st.markdown(f"#### {category}")
                for page in pages:
                    if st.button(
                        f"📚 {page['title']}", 
                        key=f"nav_{page['key']}", 
                        help=page.get("description", "View page in Confluence"),
                        use_container_width=True
                    ):
                        url = get_confluence_url(page)
                        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}">', unsafe_allow_html=True)
        
        st.sidebar.divider()
        st.sidebar.markdown("### About")
        st.sidebar.info("""
            This AI-powered assistant helps you navigate and understand 
            your Confluence documentation. Use the Page Search tool to add 
            more pages to your documentation schema.
        """)

    # Main content area
    st.title("📚 Confluence Documentation Assistant")
    st.markdown("""
        Welcome to the Documentation Assistant! This AI-powered tool helps you navigate 
        and understand your Confluence documentation. Ask questions about any of the pages 
        in your documentation schema.
    """)

    # Display available categories and their contents
    if 'schema_config' in st.session_state and st.session_state.schema_config:
        st.markdown("### Available Documentation")
        for category, pages in st.session_state.schema_config.items():
            st.markdown(f"#### {category}")
            for page in pages:
                st.markdown(f"- **{page['title']}**: {page.get('description', 'No description available')}")
    else:
        st.info("""
            No documentation pages have been added yet. Use the Page Search tool to add pages 
            to your documentation schema.
        """)

    # Chat interface
    st.divider()
    if st.button("🔄 New Chat", key="main_new_chat", use_container_width=True):
        st.session_state.messages = []
        llm_handler.clear_history()
        st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about the documentation...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Show loading state with rainbow text
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown('<p class="rainbow-text">Searching through documentation...</p>', unsafe_allow_html=True)
                
                # Process the message
                response = llm_handler.process_message(
                    prompt,
                    chat_history=st.session_state.messages[:-1]
                )
                
                # Clear the spinner and show completion message
                spinner_placeholder.markdown('<p class="rainbow-text">✨ Found relevant information!</p>', unsafe_allow_html=True)
                time.sleep(0.5)  # Brief pause to show completion message
                spinner_placeholder.empty()
                
                # Display the response
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                spinner_placeholder.empty()
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in message processing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
