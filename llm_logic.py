import logging
from typing import Annotated, Dict, TypedDict
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, Graph

# Set up logging
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]
    context: Annotated[str, "The documentation context"]
    final_answer: Annotated[str, "The final answer to return"]

def clean_html_content(html_content: str) -> str:
    """Clean HTML content and extract text."""
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)

class LLMHandler:
    def __init__(self, get_documentation_content):
        self.llm = ChatOpenAI(
            model_name="gpt-4", 
            temperature=0
        )
        self.get_documentation_content = get_documentation_content
        self.graph = self._create_graph()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()
        self.chat_history = []

    def _create_tools(self):
        """Create and return the tools for the agent."""
        def search_documentation(query: str) -> str:
            """Search through the documentation content."""
            content = self.get_documentation_content()
            if not content:
                return "No documentation content available. Please add pages to your schema first."
            return content

        doc_tool = Tool(
            name="search_documentation",
            func=search_documentation,
            description="Search through the available documentation content to find relevant information. Always use this tool first to get context before answering."
        )
        return [doc_tool]

    def _create_agent(self):
        """Create and return the agent."""
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that helps users understand documentation content that has been added to their schema.

Your primary role is to:
1. ALWAYS search the documentation first before attempting to answer
2. Only provide information that is explicitly found in the documentation
3. If the documentation doesn't contain the answer, clearly state that
4. Never make assumptions or provide generic information about topics
5. Stay focused on the specific content in the documentation

When answering:
- Start by searching the documentation
- Quote relevant sections directly when possible
- Specify which page or section the information comes from
- If information is missing or unclear, say so
- Don't provide generic information about tools or platforms
- If a question is unclear, ask for clarification

Remember: You are NOT a general knowledge assistant. You should ONLY provide information that exists in the loaded documentation content."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(self.llm, self.tools, agent_prompt)

    def _create_agent_executor(self):
        """Create and return the agent executor."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def _create_graph(self):
        """Create the LangGraph processing flow."""
        
        def process_docs(state: GraphState) -> GraphState:
            """Process documentation and add to context."""
            docs_content = self.get_documentation_content()
            if not docs_content:
                docs_content = "No documentation content available. Please add pages to your schema first."
            state["context"] = docs_content
            return state

        def analyze_content(state: GraphState) -> GraphState:
            """Analyze documentation content relative to user question."""
            messages = state.get("messages", [])
            last_message = messages[-1].content if messages else state.get("input", "")
            
            # First, check if we have any documentation content
            if not state.get("context") or state["context"] == "No documentation content available. Please add pages to your schema first.":
                state["analysis"] = "I don't have any documentation content to search through. Please add some pages to your schema first using the Page Search tool."
                return state
            
            response = self.agent_executor.invoke({
                "input": last_message,
                "chat_history": messages,
                "agent_scratchpad": []
            })
            
            state["analysis"] = response["output"]
            return state

        def format_response(state: GraphState) -> GraphState:
            """Format the final response."""
            messages = state.get("messages", [])
            last_message = messages[-1].content if messages else state.get("input", "")
            
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI assistant that helps users understand specific documentation content.
Format the analyzed information into a clear, concise response.
- Only include information found in the documentation
- Always specify which page or section information comes from
- If the answer isn't in the documentation, clearly state that
- Don't provide generic information about tools or platforms
- If the documentation content is empty, guide users to add pages"""),
                ("human", "Analysis: {analysis}\nQuestion: {question}")
            ])
            
            response = self.llm.invoke(response_prompt.format(
                analysis=state.get("analysis", ""),
                question=last_message
            ))
            
            state["final_answer"] = response.content
            return state

        # Create the graph
        workflow = Graph()
        
        workflow.add_node("process_docs", process_docs)
        workflow.add_node("analyze", analyze_content)
        workflow.add_node("format", format_response)
        
        workflow.set_entry_point("process_docs")
        workflow.add_edge("process_docs", "analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()

    def process_message(self, prompt: str, chat_history=None) -> str:
        """Process a message and return the response."""
        try:
            # Use provided chat history or instance chat history
            if chat_history is None:
                chat_history = self.chat_history
            
            # Create messages list from chat history
            messages = []
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Create state with chat history
            state = GraphState(
                messages=messages,
                context="",
                final_answer=""
            )
            
            # Process through graph
            final_state = self.graph.invoke(state)
            
            # Update chat history with new exchange
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": final_state["final_answer"]})
            
            return final_state["final_answer"]
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            raise

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []