import os
import sqlite3
from pathlib import Path
from sys import path
from dotenv import load_dotenv
import certifi
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import tools


load_dotenv()
certifi.where()
from langchain_google_genai import ChatGoogleGeminiAPI
from langchain.core.messages import SystemMessage, HumanMessage, AIMessage  
from langgraph import Graph,START,MessagesState
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.sqlite import SQLiteSaver

path("data").mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
ALLOWED_MODELS = ["gemini-2.5-flash", "gemini-1.5-turbo", "gemini-1.5-turbo-16k", "gemini-1.5-turbo-32k"]

system_promt = """You are a helpful assistant that translates natural language into SQL queries.
You will be given a natural language question and you need to generate a SQL query that can answer the question. The SQL query should be compatible with SQLite and should be able to run on the provided database schema.
You should only generate the SQL query and not provide any explanations or additional text. The SQL query should be a single line of code and should not contain any line breaks or comments. The SQL query should be optimized for performance and should return the correct results for the given question.
The database schema is as follows: 
 Rules:
- Use the table and column names exactly as they are defined in the schema.
-if the user asks about uploaded document,use search_uploaded_document function to search the document and return the result.
-use calculator for mathematical calculations and return the result.
-be clear,helpful and concise in your responses.

"""

def build_agent(model_name: str = DEFAULT_MODEL):
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Model {model_name} is not allowed. Allowed models are: {ALLOWED_MODELS}")

    llm = ChatGoogleGeminiAPI(model=model_name, temperature=0.7, api_key=os.environ.get("GOOGLE_API_KEY"))
    llm_with_tools= llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = state.get_messages()
        response = llm_with_tools(messages)
        state.add_message(AIMessage(content=response.content))
        return state
    tool_node = ToolNode(tools)

    workflow = Graph(message_state=MessagesState(), nodes=[START, tool_node, chatbot_node], edges=[(START, tool_node), (tool_node, chatbot_node)])