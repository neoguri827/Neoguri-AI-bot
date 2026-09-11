import json
import time
import secrets
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


class UsageStore:
    """봇별 토큰 사용량을 날짜(KST) 단위로 Upstash에 누적 기록한다.
    재시작이 잦아도(Render 무료 플랜) 집계가 사라지지 않도록 메모리 대신 여기에 쌓는다."""

    def __init__(self, redis_url: str, redis_token: str, namespace: str = "usage"):
        self.redis = Redis(url=redis_url, token=redis_token)
        self.namespace = namespace

    def _key(self, date_str: str) -> str:
        return f"{self.namespace}:{date_str}"

    def add(self, date_str: str, bot_name: str, total_tokens: int):
        self.redis.hincrby(self._key(date_str), bot_name, total_tokens)

    def get_day(self, date_str: str) -> Dict[str, int]:
        raw = self.redis.hgetall(self._key(date_str)) or {}
        return {k: int(v) for k, v in raw.items()}

    def get_range(self, date_strs: List[str]) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for date_str in date_strs:
            for bot_name, count in self.get_day(date_str).items():
                totals[bot_name] = totals.get(bot_name, 0) + count
        return totals


class ExpenseStore:
    """채팅방(chat_id)별 지출 기록을 Upstash 해시에 저장한다.
    필드 키는 밀리초 타임스탬프라 자동으로 고유하고, 조회 시 자연스럽게 최신순 정렬이 가능하다."""

    def __init__(self, redis_url: str, redis_token: str, namespace: str = "expense"):
        self.redis = Redis(url=redis_url, token=redis_token)
        self.namespace = namespace

    def _key(self, chat_id: int) -> str:
        return f"{self.namespace}:{chat_id}"

    def add(self, chat_id: int, record: Dict) -> str:
        # 밀리초 타임스탬프만으로는 같은 순간에 두 건이 들어오면 충돌해서 하나가 덮어써질 수
        # 있으므로(예: 여러 파일을 빠르게 연달아 보낼 때), 짧은 랜덤 접미사로 유일성을 보장한다.
        entry_id = f"{int(time.time() * 1000)}-{secrets.token_hex(3)}"
        self.redis.hset(self._key(chat_id), entry_id, json.dumps(record, ensure_ascii=False))
        return entry_id

    def remove(self, chat_id: int, entry_id: str) -> bool:
        removed = self.redis.hdel(self._key(chat_id), entry_id)
        return bool(removed)

    def list_all(self, chat_id: int) -> List[Dict]:
        raw = self.redis.hgetall(self._key(chat_id)) or {}
        entries = []
        for entry_id, value in raw.items():
            try:
                record = json.loads(value)
            except Exception:
                continue
            record["id"] = entry_id
            entries.append(record)
        return entries

    def clear(self, chat_id: int):
        self.redis.delete(self._key(chat_id))


class AlarmScheduleStore:
    """알람봇이 재배포·재시작을 겪어도 같은 시간대에 브리핑을 중복 발송하지 않도록
    마지막으로 발송한 시간대(한국시간 기준 'YYYY-MM-DD HH')를 기억한다."""

    def __init__(self, redis_url: str, redis_token: str, namespace: str):
        self.redis = Redis(url=redis_url, token=redis_token)
        self.key = f"{namespace}:last_sent_hour"

    def get_last_sent_hour(self) -> Optional[str]:
        return self.redis.get(self.key)

    def set_last_sent_hour(self, hour_slot: str):
        self.redis.set(self.key, hour_slot)
