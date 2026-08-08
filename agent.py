import os
from pathlib import Path

from dotenv import load_dotenv
import certifi

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import tools

Path("data").mkdir(exist_ok=True)


# Update default and allowed models to use Groq chat models
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ALLOWED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}



SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named ChiruGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""



def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name




def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected Groq model.
    """

    selected_model = normalize_model_name(model_name)

    llm = ChatGroq(
        model=selected_model,
        temperature=0.3,
        streaming=True
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]