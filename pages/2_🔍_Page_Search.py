import streamlit as st
import base64
import os
import requests
import logging
from typing import Optional, Dict, List
import json
import textwrap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Confluence Page Search",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for API configuration
if "confluence_api_token" not in st.session_state:
    st.session_state.confluence_api_token = ""
if "confluence_email" not in st.session_state:
    st.session_state.confluence_email = ""
if "confluence_url" not in st.session_state:
    st.session_state.confluence_url = ""

# Sidebar API Configuration
with st.sidebar:
    st.divider()
    st.markdown("### 🔑 API Configuration")
    
    # Confluence API configuration
    confluence_url = st.text_input(
        "Confluence URL",
        value=st.session_state.confluence_url,
        help="Enter your Confluence instance URL (e.g., https://your-company.atlassian.net/wiki)",
        key="confluence_url_input"
    )
    
    confluence_email = st.text_input(
        "Confluence Email",
        value=st.session_state.confluence_email,
        help="Enter your Confluence account email",
        key="confluence_email_input"
    )
    
    confluence_token = st.text_input(
        "Confluence API Token",
        type="password",
        value=st.session_state.confluence_api_token,
        help="Enter your Confluence API token. Generate one in your Confluence settings.",
        key="confluence_token_input"
    )
    
    if st.button("Save API Configuration", type="primary"):
        if confluence_email and confluence_token and confluence_url:
            st.session_state.confluence_email = confluence_email
            st.session_state.confluence_api_token = confluence_token
            st.session_state.confluence_url = confluence_url.rstrip('/')
            st.success("✅ API configuration saved!")
            st.rerun()
        else:
            st.error("Please fill in all API configuration fields")

# Check for required API configuration
if not (st.session_state.confluence_api_token and 
        st.session_state.confluence_email and 
        st.session_state.confluence_url):
    st.error("⚠️ Please configure your Confluence API credentials in the sidebar")
    st.stop()

# Confluence API configuration
BASE_URL = f"{st.session_state.confluence_url}/api/v2"
EMAIL = st.session_state.confluence_email
API_TOKEN = st.session_state.confluence_api_token
auth_header = {"Authorization": "Basic " + base64.b64encode(f"{EMAIL}:{API_TOKEN}".encode()).decode()}

# Initialize session state
if "page_cursor" not in st.session_state:
    st.session_state.page_cursor = None
if "current_pages" not in st.session_state:
    st.session_state.current_pages = []
if "all_loaded_pages" not in st.session_state:
    st.session_state.all_loaded_pages = []
if "selected_space" not in st.session_state:
    st.session_state.selected_space = None
if "space_changed" not in st.session_state:
    st.session_state.space_changed = False
if "schema_config" not in st.session_state:
    st.session_state.schema_config = {}

# Function to handle space selection
def on_space_change():
    """Handle space selection change."""
    st.session_state.page_cursor = None
    st.session_state.current_pages = []
    st.session_state.all_loaded_pages = []
    st.session_state.space_changed = True

# Function to load more pages
def load_more_pages():
    """Load pages for the selected space."""
    try:
        space_id = st.session_state.selected_space["id"] if st.session_state.selected_space else None
        new_pages = get_space_pages(space_id=space_id, cursor=st.session_state.page_cursor)
        
        if new_pages.get("results"):
            st.session_state.current_pages = new_pages.get("results", [])
            st.session_state.all_loaded_pages.extend(st.session_state.current_pages)
            
            # Update cursor for next page
            next_link = new_pages.get("_links", {}).get("next", "")
            st.session_state.page_cursor = next_link.split("cursor=")[-1] if "cursor=" in next_link else None
            
        st.session_state.space_changed = False
    except Exception as e:
        st.error(f"Error loading pages: {str(e)}")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_spaces() -> List[Dict]:
    """Fetch all active Confluence spaces."""
    try:
        params = {
            "status": "current",  # Only get active spaces
            "type": "global",     # Only get global spaces
            "sort": "name",       # Sort by name
            "limit": 250          # Get more at once
        }
        
        response = requests.get(f"{BASE_URL}/spaces", headers=auth_header, params=params)
        
        if response.status_code == 401:
            st.error("Authentication failed. Please check your Confluence email and API token.")
            return []
        elif response.status_code == 403:
            st.error("Access denied. Please check your Confluence permissions.")
            return []
        elif response.status_code == 404:
            st.error("API endpoint not found. Please check your Confluence instance URL.")
            return []
            
        response.raise_for_status()
        
        result = response.json()
        spaces = result.get("results", [])
        
        # Sort spaces by name for better UX
        sorted_spaces = sorted(spaces, key=lambda x: x["name"].lower())
        return sorted_spaces
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to Confluence. Please check your internet connection.")
        return []
    except Exception as e:
        logger.error(f"Error fetching spaces: {str(e)}")
        st.error(f"Error loading spaces: {str(e)}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_space_pages(space_id: Optional[str] = None) -> List[Dict]:
    """Fetch all pages from a space."""
    try:
        params = {
            "limit": 250,  # Maximum limit to get all pages at once
            "status": "current"
        }
        
        # Add space ID if provided
        if space_id:
            params["space-id"] = [space_id]
        
        response = requests.get(f"{BASE_URL}/pages", headers=auth_header, params=params)
        
        if response.status_code == 401:
            st.error("Authentication failed. Please check your Confluence email and API token.")
            return []
        elif response.status_code == 403:
            st.error("Access denied. Please check your Confluence permissions.")
            return []
        elif response.status_code == 404:
            st.error("API endpoint not found. Please check your Confluence instance URL.")
            return []
            
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to Confluence. Please check your internet connection.")
        return []
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []

def get_page_details(page_id: str) -> Optional[Dict]:
    """Get detailed information about a specific page."""
    try:
        response = requests.get(f"{BASE_URL}/pages/{page_id}", headers=auth_header)
        
        if response.status_code == 401:
            st.error("Authentication failed. Please check your Confluence email and API token.")
            return None
        elif response.status_code == 403:
            st.error("Access denied. Please check your Confluence permissions.")
            return None
        elif response.status_code == 404:
            st.error("Page not found. It may have been deleted or moved.")
            return None
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to Confluence. Please check your internet connection.")
        return None
    except Exception as e:
        logger.error(f"Error getting page details: {str(e)}")
        return None

# Page header
st.title("🔍 Confluence Page Browser")
st.markdown("""
    Browse active Confluence spaces and pages to get configuration details.
    Select a space to view its pages and get their IDs.
""")

# Fetch spaces with loading indicator
with st.status("Loading active spaces...", expanded=False) as status:
    spaces = get_spaces()
    if spaces:
        status.update(label=f"✅ Found {len(spaces)} active spaces", state="complete")
    else:
        status.update(label="❌ Failed to load spaces", state="error")

# Space and page selection interface
col1, col2 = st.columns([1, 2])

selected_space = None
with col1:
    st.subheader("1. Select Space")
    if spaces:
        space_options = ["All Spaces"] + [
            f"{space['name']} ({space['key']})" 
            for space in spaces
        ]
        
        # Store previous selection to detect changes
        previous_space = st.session_state.get("selected_space")
        
        selected_space_name = st.selectbox(
            "Choose a space",
            options=space_options,
            help="Select a Confluence space to browse its pages",
            key="space_selector"
        )
        
        # Update selected space
        if selected_space_name != "All Spaces":
            st.session_state.selected_space = next(
                space for space in spaces 
                if f"{space['name']} ({space['key']})" == selected_space_name
            )
        else:
            st.session_state.selected_space = None
        
        # Check if space changed
        if previous_space != st.session_state.selected_space:
            st.session_state.page_cursor = None
            st.session_state.current_pages = []
            st.session_state.all_loaded_pages = []
            st.session_state.space_changed = True
            st.rerun()  # Force rerun to update pages immediately

with col2:
    st.subheader("2. Browse Pages")
    if spaces:
        # Load pages for selected space
        with st.spinner("Loading pages..."):
            pages = get_space_pages(
                space_id=st.session_state.selected_space["id"] 
                if st.session_state.selected_space else None
            )
        
        if pages:
            st.success(f"Found {len(pages)} pages")
            
            # Format page options
            page_options = [
                f"{page['title']} (ID: {page['id']})" 
                for page in pages
            ]
            
            # Page selection
            selected_page = st.selectbox(
                "Select a page to view details",
                options=page_options,
                help="Choose a page to see its configuration details",
                key="page_selector"
            )
            
            # Show page details if selected
            if selected_page:
                page_id = selected_page.split("ID: ")[-1].rstrip(")")
                page_details = get_page_details(page_id)
                
                if page_details:
                    st.markdown("### 📄 Page Details")
                    
                    # Display page information
                    st.markdown(f"""
                    - **Title**: {page_details['title']}
                    - **ID**: `{page_details['id']}`
                    - **Space**: {page_details['spaceId']}
                    - **Created**: {page_details['createdAt'][:10]}
                    - **Last Updated**: {page_details['version']['createdAt'][:10]}
                    """)
                    
                    st.divider()
                    
                    # Add to Category section
                    st.markdown("### 📚 Add to Documentation")
                    
                    # Category selection/creation
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Get existing categories
                        categories = list(st.session_state.schema_config.keys())
                        
                        if categories:
                            selected_category = st.selectbox(
                                "Choose existing category",
                                options=[""] + categories,
                                key="existing_category"
                            )
                        else:
                            selected_category = ""
                            st.info("No categories exist yet. Create one below.")
                    
                    with col2:
                        new_category = st.text_input("Or create new category", key="new_category_page_details")
                        if st.button("Create Category", key="create_category_page_details") and new_category:
                            if new_category not in st.session_state.schema_config:
                                st.session_state.schema_config[new_category] = []
                                selected_category = new_category
                                st.success(f"Created category: {new_category}")
                                st.rerun()
                            else:
                                st.warning("Category already exists!")
                    
                    # Add page to category
                    if selected_category or new_category:
                        target_category = selected_category or new_category
                        
                        # Create page config
                        if st.session_state.selected_space and page_details:
                            page_config = {
                                "space": st.session_state.selected_space['key'],
                                "page_id": page_details['id'],
                                "title": page_details['title'],
                                "description": "Add description here",
                                "key": page_details['title'].lower().replace(' ', '_')
                            }
                            
                            # Optional description
                            page_config["description"] = st.text_area(
                                "Page description (optional)",
                                value=page_config["description"],
                                help="Add a brief description of what this page contains"
                            )
                            
                            # Add to category button
                            if st.button("📝 Add to Documentation", type="primary", use_container_width=True):
                                # Check if page already exists in category
                                if not any(p['page_id'] == page_config['page_id'] 
                                         for p in st.session_state.schema_config.get(target_category, [])):
                                    if target_category not in st.session_state.schema_config:
                                        st.session_state.schema_config[target_category] = []
                                    st.session_state.schema_config[target_category].append(page_config)
                                    st.success(f"Added '{page_details['title']}' to {target_category}!")
                                else:
                                    st.warning("This page is already in the selected category!")
                        else:
                            st.error("Please select a space and page first.")
                    
                    st.divider()
                    st.link_button("🔗 Open in Confluence", page_details['_links']['webui'])
        else:
            st.warning("No pages found in the selected space.")

# Help section
with st.sidebar:
    st.markdown("### 🔍 Quick Tips")
    st.info("""
        1. Choose a space from the dropdown
        2. Select from recent pages
        3. Copy page ID or full configuration
        4. View page details and URL
    """)
    
    if st.session_state.selected_space:
        st.markdown("### 📊 Space Stats")
        st.success(f"""
        **{st.session_state.selected_space['name']}**
        - Type: {st.session_state.selected_space['type']}
        - Key: `{st.session_state.selected_space['key']}`
        - Homepage: {st.session_state.selected_space.get('homepageId', 'N/A')}
        - Created: {st.session_state.selected_space['createdAt'][:10]}
        """) 

# Add after the main page content, before the sidebar
st.divider()
st.header("📝 Documentation Schema Builder")
st.markdown("""
    Build your documentation schema by exploring pages and adding them to categories.
    The schema will be in the format used by the Chat Assistant.
""")

# Schema builder section
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Add to Schema")
    
    # Category management
    new_category = st.text_input("New Category Name", key="new_category_schema_builder")
    if st.button("Add Category", key="add_category_schema_builder") and new_category:
        if new_category not in st.session_state.schema_config:
            st.session_state.schema_config[new_category] = []
            st.success(f"Added category: {new_category}")
        else:
            st.warning("Category already exists!")

    # Add current page to category
    if page_details and st.session_state.selected_space:
        st.markdown("### Add Current Page")
        st.markdown(f"**Selected:** {page_details['title']}")
        
        categories = list(st.session_state.schema_config.keys())
        if categories:
            selected_category = st.selectbox(
                "Choose category",
                options=categories,
                key="category_selector"
            )
            
            if st.button("Add to Category", key="add_to_category"):
                if st.session_state.selected_space:  # Add check here
                    page_config = {
                        "space": st.session_state.selected_space['key'],
                        "page_id": page_details['id'],
                        "title": page_details['title'],
                        "description": "Add description here",
                        "key": page_details['title'].lower().replace(' ', '_')
                    }
                    
                    # Check if page already exists in category
                    if not any(p['page_id'] == page_config['page_id'] 
                             for p in st.session_state.schema_config[selected_category]):
                        st.session_state.schema_config[selected_category].append(page_config)
                        st.success("Added to schema!")
                    else:
                        st.warning("Page already in category!")
                else:
                    st.error("Please select a space first.")
        else:
            st.warning("Create a category first!")

with col2:
    st.subheader("Current Schema")
    
    # Show current schema
    if st.session_state.schema_config:
        # Add schema management buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy Schema"):
                st.code(json.dumps(st.session_state.schema_config, indent=4))
                st.success("Schema copied!")
        with col2:
            if st.button("💾 Save Schema"):
                st.download_button(
                    label="Download Schema",
                    data=json.dumps(st.session_state.schema_config, indent=4),
                    file_name="confluence_schema.json",
                    mime="application/json"
                )
        with col3:
            if st.button("🗑️ Clear Schema"):
                st.session_state.schema_config = {}
                st.rerun()
        
        # Display current schema
        for category, pages in st.session_state.schema_config.items():
            with st.expander(f"📚 {category}", expanded=True):
                st.markdown(f"**{len(pages)} pages**")
                for page in pages:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"- {page['title']}")
                    with col2:
                        if st.button("🗑️", key=f"remove_{category}_{page['page_id']}"):
                            st.session_state.schema_config[category].remove(page)
                            st.rerun()
    else:
        st.info("Start building your schema by adding categories and pages!") 