import os
from pathlib import Path

from dotenv import load_dotenv
import certifi

from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import tools

Path("data").mkdir(exist_ok=True)


# Update default and allowed models to use Groq chat models
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

ALLOWED_MODELS = {
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}



SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named ChiruGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents when the user asks about files they uploaded.
4. Search the web for latest/current information.
5. Remember important user information.
6. Recall saved memory when useful.
7. Do math calculations.

Rules:
- Use the most appropriate available tool when a tool is needed.
- When using retrieved tool output, answer directly from that output.
- Do not invent function-call syntax in plain text.
- If you already received tool results in the current turn, answer directly from those results and do not call the same tool again.
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

        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            response = llm.invoke(messages)
        else:
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