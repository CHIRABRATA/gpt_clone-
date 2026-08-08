from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import Lock


Path("data").mkdir(exist_ok=True)

STORE_PATH = Path("data/chatbot_memory.json")
_STORE_LOCK = Lock()


@dataclass(slots=True)
class Conversation:
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class ChatMessage:
    thread_id: str
    role: str
    content: str
    created_at: datetime


@dataclass(slots=True)
class LongTermMemory:
    thread_id: str
    memory: str
    created_at: datetime


def _empty_store() -> dict:
    return {
        "conversations": [],
        "chat_messages": [],
        "long_term_memory": []
    }


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_store() -> dict:
    if not STORE_PATH.exists():
        return _empty_store()

    try:
        with STORE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _empty_store()

    store = _empty_store()

    for key in store:
        items = data.get(key, [])
        if isinstance(items, list):
            store[key] = items

    return store


def _save_store(store: dict) -> None:
    temp_path = STORE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False, indent=2)

    temp_path.replace(STORE_PATH)


def _conversation_from_dict(item: dict) -> Conversation:
    return Conversation(
        thread_id=item["thread_id"],
        title=item["title"],
        created_at=_parse_datetime(item["created_at"]),
        updated_at=_parse_datetime(item["updated_at"])
    )


def _message_from_dict(item: dict) -> ChatMessage:
    return ChatMessage(
        thread_id=item["thread_id"],
        role=item["role"],
        content=item["content"],
        created_at=_parse_datetime(item["created_at"])
    )


def _memory_from_dict(item: dict) -> LongTermMemory:
    return LongTermMemory(
        thread_id=item["thread_id"],
        memory=item["memory"],
        created_at=_parse_datetime(item["created_at"])
    )


def init_db():
    with _STORE_LOCK:
        if not STORE_PATH.exists():
            _save_store(_empty_store())


def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    with _STORE_LOCK:
        store = _load_store()
        conversations = store["conversations"]
        now = datetime.utcnow().isoformat()

        for conversation in conversations:
            if conversation["thread_id"] == thread_id:
                conversation["updated_at"] = now
                _save_store(store)
                return

        title = "New Chat"

        if first_message:
            cleaned_message = first_message.strip()
            if cleaned_message:
                title = cleaned_message[:40]
                if len(cleaned_message) > 40:
                    title += "..."

        conversations.append(
            {
                "thread_id": thread_id,
                "title": title,
                "created_at": now,
                "updated_at": now
            }
        )
        _save_store(store)


def list_conversations():
    with _STORE_LOCK:
        store = _load_store()
        conversations = [
            _conversation_from_dict(item)
            for item in store["conversations"]
        ]

        return sorted(conversations, key=lambda item: item.updated_at, reverse=True)


def save_chat_message(thread_id: str, role: str, content: str):
    with _STORE_LOCK:
        store = _load_store()
        now = datetime.utcnow().isoformat()

        store["chat_messages"].append(
            {
                "thread_id": thread_id,
                "role": role,
                "content": content,
                "created_at": now
            }
        )

        for conversation in store["conversations"]:
            if conversation["thread_id"] == thread_id:
                conversation["updated_at"] = now
                break

        _save_store(store)


def get_chat_history(thread_id: str):
    with _STORE_LOCK:
        store = _load_store()
        messages = [
            _message_from_dict(item)
            for item in store["chat_messages"]
            if item["thread_id"] == thread_id
        ]

        return sorted(messages, key=lambda item: item.created_at)


def save_memory(thread_id: str, memory: str):
    with _STORE_LOCK:
        store = _load_store()

        store["long_term_memory"].append(
            {
                "thread_id": thread_id,
                "memory": memory,
                "created_at": datetime.utcnow().isoformat()
            }
        )

        _save_store(store)
        return "Memory saved successfully."


def search_memory(thread_id: str, query: str):
    with _STORE_LOCK:
        store = _load_store()
        memories = [
            _memory_from_dict(item)
            for item in store["long_term_memory"]
            if item["thread_id"] == thread_id
        ]

        if not memories:
            return "No saved memory found."

        query = query.strip().lower()
        if query:
            matched_memories = [
                memory for memory in memories
                if query in memory.memory.lower()
            ]
            if matched_memories:
                memories = matched_memories

        recent_memories = sorted(
            memories,
            key=lambda item: item.created_at,
            reverse=True
        )[:20]

        return "\n".join([f"- {memory.memory}" for memory in recent_memories])