import json
from typing import Dict, List, Optional
from upstash_redis import Redis


class ChatHistoryStore:
    def __init__(self, redis_url: str, redis_token: str, namespace: str):
        self.redis = Redis(url=redis_url, token=redis_token)
        self.namespace = namespace

    def _key(self, chat_id: int) -> str:
        return f"{self.namespace}:chat:{chat_id}"

    def load_history(self, chat_id: int, limit: int = 12) -> List[Dict[str, str]]:
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

    def get_relevant_excerpt(self, chat_id: int, name: str, keywords: List[str], max_chars: int) -> Optional[str]:
        """문서 전체 대신, 질문 키워드와 관련된 문단만 최대 max_chars 이내로 발췌해서 반환한다."""
        content = self.get(chat_id, name)
        if content is None:
            return None
        if len(content) <= max_chars:
            return content

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content]

        lowered_keywords = [k.lower() for k in keywords if k]
        scored = []
        for idx, para in enumerate(paragraphs):
            para_lower = para.lower()
            score = sum(para_lower.count(k) for k in lowered_keywords) if lowered_keywords else 0
            scored.append((score, idx, para))

        relevant = sorted((p for p in scored if p[0] > 0), key=lambda x: (-x[0], x[1]))
        selected: Dict[int, str] = {}
        total_len = 0
        for score, idx, para in relevant:
            if total_len + len(para) > max_chars:
                continue
            selected[idx] = para
            total_len += len(para)

        if not selected:
            excerpt = content[:max_chars]
            return excerpt + f"\n...(문서 앞부분 {max_chars:,}자만 표시, 전체 {len(content):,}자 중 일부)"

        ordered_idx = sorted(selected.keys())
        excerpt = "\n\n".join(selected[i] for i in ordered_idx)
        if len(selected) < len(paragraphs):
            excerpt += f"\n...(질문과 관련된 문단만 발췌함, 전체 {len(content):,}자 중 {len(excerpt):,}자 표시)"
        return excerpt
