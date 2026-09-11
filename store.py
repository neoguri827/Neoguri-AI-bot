import json
from typing import Dict, List, Optional
from upstash_redis import Redis


class ChatHistoryStore:
    def __init__(self, redis_url: str, redis_token: str, namespace: str):
        self.redis = Redis(url=redis_url, token=redis_token)
        self.namespace = namespace

    def _key(self, chat_id: int) -> str:
        return f"{self.namespace}:chat:{chat_id}"

    def load_history(self, chat_id: int, limit: int = 20) -> List[Dict[str, str]]:
        key = self._key(chat_id)
        raw_items = self.redis.lrange(key, -limit, -1)
        history = []
        for item in raw_items:
            try:
                history.append(json.loads(item))
            except Exception:
                continue
        return history

    def append(self, chat_id: int, role: str, content: str, max_rows: int = 200):
        key = self._key(chat_id)
        self.redis.rpush(key, json.dumps({"role": role, "content": content}))
        self.redis.ltrim(key, -max_rows, -1)

    def clear(self, chat_id: int):
        self.redis.delete(self._key(chat_id))


class KnowledgeStore:
    """대화 리셋이나 재배포에도 사라지지 않는 영구 참고자료 저장소"""

    def __init__(self, redis_url: str, redis_token: str, namespace: str):
        self.redis = Redis(url=redis_url, token=redis_token)
        self.namespace = namespace

    def _key(self, chat_id: int) -> str:
        return f"{self.namespace}:kb:{chat_id}"

    def add(self, chat_id: int, name: str, content: str):
        self.redis.hset(self._key(chat_id), name, content)

    def get(self, chat_id: int, name: str) -> Optional[str]:
        return self.redis.hget(self._key(chat_id), name)

    def remove(self, chat_id: int, name: str) -> bool:
        removed = self.redis.hdel(self._key(chat_id), name)
        return bool(removed)

    def list_names(self, chat_id: int) -> List[str]:
        return list(self.redis.hkeys(self._key(chat_id)) or [])

    def get_all_text(self, chat_id: int) -> str:
        data = self.redis.hgetall(self._key(chat_id)) or {}
        if not data:
            return ""
        parts = [f"[참고자료: {name}]\n{content}" for name, content in data.items()]
        return "\n\n".join(parts)
