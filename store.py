import json
from typing import Dict, List
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
