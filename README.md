# Confluence Documentation Assistant 🤖

An AI-powered documentation assistant that helps teams navigate and understand their Confluence content. Built with Streamlit and powered by GPT-4, this tool allows you to create a curated collection of Confluence pages and interact with their content through natural language.

## 🌟 Features

- **Smart Documentation Search**: Ask questions about your Confluence pages in natural language
- **Custom Categories**: Organize pages into meaningful categories for better navigation
- **Interactive UI**: Easy-to-use interface for managing documentation and asking questions
- **Page Management**: Add, remove, and organize Confluence pages with ease
- **Schema Builder**: Create and manage documentation schemas that can be shared across teams
- **Real-time Updates**: Changes to your schema are reflected immediately in the chat interface
- **Simple Configuration**: Set up API keys directly in the UI without environment variables

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- Confluence API token (generate in your Confluence settings)
- Access to Confluence spaces you want to query

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/confluence-chat.git
cd confluence-chat
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run Home.py
```

### Configuration

1. Once the application is running, look for the API Configuration section in the sidebar
2. Enter your:
   - OpenAI API key
   - Confluence API token
   - Confluence email address
3. Click "Save API Configuration"
4. Your credentials will be securely stored in the session

## 📚 Usage Guide

### 1. Page Search Tool (🔍)

The Page Search tool allows you to browse and add Confluence pages to your documentation schema:

1. Select a Confluence space from the dropdown
2. Browse available pages in that space
3. View page details and add them to your schema
4. Organize pages into categories
5. Add optional descriptions for better context

### 2. Chat Assistant (💬)

The Chat Assistant helps you interact with your documentation:

1. Ask questions about any pages in your schema
2. Get relevant information with source references
3. Navigate directly to source pages
4. Start new conversations when needed
5. View your chat history

### 3. Schema Management

Manage your documentation schema:

- Create new categories
- Add pages to categories
- Remove pages when needed
- Export your schema as JSON
- Import existing schemas

## 🔧 Configuration

### Customizing the Assistant

The assistant's behavior can be customized by modifying the prompts in `llm_logic.py`:

- Adjust the system prompt to change how the assistant interprets questions
- Modify the response format for different output styles
- Configure the maximum number of iterations for complex queries

### Performance Tuning

Several caching mechanisms are in place:

- `@st.cache_data`: For API responses (spaces, pages)
- `@st.cache_resource`: For LLM initialization and documentation content

## 🛠️ Architecture

The application consists of three main components:

1. **Page Search (`pages/2_🔍_Page_Search.py`)**
   - Handles Confluence API interactions
   - Manages page discovery and schema building
   - Provides the UI for page management

2. **Chat Assistant (`pages/1_💬_Chat_Assistant.py`)**
   - Manages chat interface
   - Handles user interactions
   - Displays documentation content

3. **LLM Logic (`llm_logic.py`)**
   - Processes natural language queries
   - Manages context and chat history
   - Formats responses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI GPT-4](https://openai.com/)
- Uses [LangChain](https://langchain.com/) for LLM interactions
- Integrates with [Confluence Cloud REST API](https://developer.atlassian.com/cloud/confluence/rest/v2/intro/)

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Make sure to enter valid API keys in the sidebar configuration
   - Check that your OpenAI API key has sufficient credits
   - Verify your Confluence API token has the necessary permissions

2. **Confluence Access**
   - Ensure your Confluence email has access to the spaces you want to query
   - Check space and page access rights
   - Verify your API token hasn't expired

3. **Performance Issues**
   - Clear Streamlit cache if pages aren't updating
   - Restart the application if responses become slow
   - Check your internet connection for API access

### Getting Help

- Open an issue for bugs or feature requests
- Check existing issues for solutions
- Contact the maintainers for serious issues

## 🔜 Roadmap

Future improvements planned:

- [ ] Multi-user support
- [ ] Advanced search capabilities
- [ ] Custom prompt templates
- [ ] Bulk page import
- [ ] Schema sharing features
- [ ] Integration with other documentation platforms
- [ ] Persistent API key storage (optional)

## 📊 Status

Project is: _in active development_

Last updated: [Current Date] 