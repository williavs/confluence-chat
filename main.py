import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Confluence Documentation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-left: 4px solid #0070D4;
    }
    .security-note {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e7f5ff;
        border: 1px solid #0070D4;
        margin: 1rem 0;
    }
    .step-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
        border: 1px solid #dee2e6;
    }
    h1, h2, h3 {
        color: #1E1E1E;
    }
    .highlight {
        color: #0070D4;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 Confluence Documentation Assistant")
st.markdown("### Your AI-Powered Guide to Confluence Documentation")

# Introduction
st.markdown("""
Welcome to the Confluence Documentation Assistant! This powerful tool helps teams navigate and understand their 
Confluence documentation through natural language conversations and smart organization.
""")

# Key Features Section
st.header("🌟 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Smart Search</h3>
        <p>Ask questions in natural language and get precise answers from your documentation.</p>
    </div>
    
    <div class="feature-card">
        <h3>📚 Custom Categories</h3>
        <p>Organize your documentation into meaningful categories for better navigation and understanding.</p>
    </div>
    
    <div class="feature-card">
        <h3>🤖 AI-Powered</h3>
        <p>Powered by GPT-4 for intelligent understanding and natural conversations about your documentation.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔐 Secure Access</h3>
        <p>Your API keys are stored securely in session state and never persisted or shared.</p>
    </div>
    
    <div class="feature-card">
        <h3>📋 Schema Builder</h3>
        <p>Create and manage documentation schemas that can be shared across your team.</p>
    </div>
    
    <div class="feature-card">
        <h3>⚡ Real-time Updates</h3>
        <p>Changes to your schema are reflected immediately in the chat interface.</p>
    </div>
    """, unsafe_allow_html=True)

# Security Section
st.header("🔒 Security First")
st.markdown("""
<div class="security-note">
    <h3>Secure Credential Management</h3>
    <p>We take security seriously. Here's how we protect your credentials:</p>
    <ul>
        <li><strong>Session-Only Storage:</strong> API keys are stored only in your browser's session state</li>
        <li><strong>No Persistence:</strong> Credentials are never saved to disk or databases</li>
        <li><strong>Automatic Clearing:</strong> All credentials are cleared when you close your browser</li>
        <li><strong>Encrypted Transit:</strong> All API communications use HTTPS encryption</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Getting Started Section
st.header("🚀 Getting Started")

st.markdown("""
<div class="step-card">
    <h3>1️⃣ Configure Your API Keys</h3>
    <ul>
        <li>Open the Page Search (🔍) or Chat Assistant (💬) tool</li>
        <li>Look for the API Configuration section in the sidebar</li>
        <li>Enter your Confluence email and API token</li>
        <li>Click "Save API Configuration"</li>
    </ul>
</div>

<div class="step-card">
    <h3>2️⃣ Build Your Documentation Schema</h3>
    <ul>
        <li>Use the Page Search tool to browse your Confluence spaces</li>
        <li>Create categories for organizing your documentation</li>
        <li>Add relevant pages to your categories</li>
        <li>Add descriptions to help understand page content</li>
    </ul>
</div>

<div class="step-card">
    <h3>3️⃣ Start Chatting</h3>
    <ul>
        <li>Switch to the Chat Assistant</li>
        <li>Ask questions about your documentation</li>
        <li>Get precise answers with source references</li>
        <li>Navigate directly to source pages when needed</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# How It Works Section
st.header("⚙️ How It Works")

st.markdown("""
The Confluence Documentation Assistant combines several powerful technologies:

1. **Confluence Integration**
   - Direct connection to your Confluence instance
   - Real-time access to your documentation
   - Secure API communication

2. **AI Processing**
   - GPT-4 for natural language understanding
   - Context-aware responses
   - Smart document categorization

3. **Schema Management**
   - Flexible documentation organization
   - JSON-based schema format
   - Easy sharing and version control
""")

# Tips Section
st.header("💡 Pro Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **For Better Results:**
    - Be specific in your questions
    - Organize related pages in categories
    - Add descriptive page summaries
    - Use the schema builder regularly
    """)

with col2:
    st.markdown("""
    **Best Practices:**
    - Update your schema as docs change
    - Share schemas with your team
    - Use natural language questions
    - Check source pages for context
    """)

# Built With Section
st.header("🛠️ Built With")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Frontend**
    - Streamlit
    - Custom CSS
    - Responsive Design
    """)

with col2:
    st.markdown("""
    **Backend**
    - Python
    - LangChain
    - OpenAI GPT-4
    """)

with col3:
    st.markdown("""
    **APIs**
    - Confluence Cloud REST API
    - OpenAI API
    - Custom Integrations
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center">
    <p>Built with ❤️ for documentation lovers everywhere</p>
    <p>Need help? Check out our <a href="https://github.com/williavs/confluence-chat">GitHub repository</a></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Quick Links")
    st.markdown("""
    - [Page Search Tool](Page_Search)
    - [Chat Assistant](Chat_Assistant)
    - [GitHub Repository](https://github.com/williavs/confluence-chat)
    """)
    
    st.divider()
    st.markdown("### 🎯 Get Started")
    if st.button("🔍 Go to Page Search", use_container_width=True):
        st.switch_page("pages/2_🔍_Page_Search.py")
    if st.button("💬 Open Chat Assistant", use_container_width=True):
        st.switch_page("pages/1_💬_Chat_Assistant.py") 