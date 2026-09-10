import os
import time
import logging
import threading
from typing import List
from google import genai

logger = logging.getLogger(__name__)


def resolve_api_key(specific_env: str) -> str:
    key = os.environ.get(specific_env) or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(f"{specific_env} 또는 GEMINI_API_KEY 환경변수가 필요합니다.")
    return key


class GeminiRouter:
    FALLBACK_MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    EXCLUDE_KEYWORDS = [
        "tts", "audio", "image", "vision", "embedding",
        "aqa", "live", "veo", "imagen", "learnlm", "gemma",
    ]

    MAX_MODEL_SWITCHES_PER_CALL = 6

    def __init__(self, client: genai.Client, label: str = ""):
        self.client = client
        self.label = label
        self._lock = threading.Lock()
        self.models = self._discover_models() or list(self.FALLBACK_MODELS)
        self.model_index = 0
        self.model_name = self.models[0]
        logger.info(f"[{self.label}] 사용 가능한 모델 {len(self.models)}개 확인, 1순위로 시작: {self.model_name}")

    def _discover_models(self) -> List[str]:
        try:
            raw_models = list(self.client.models.list())
        except Exception as e:
            logger.warning(f"[{self.label}] 모델 목록 조회 실패, 기본 후보 목록 사용: {e}")
            return []

        usable = []
        for m in raw_models:
            name = getattr(m, "name", None)
            if not name:
                continue
            short_name = name.split("/")[-1]
            if any(k in short_name.lower() for k in self.EXCLUDE_KEYWORDS):
                continue
            supported = (
                getattr(m, "supported_actions", None)
                or getattr(m, "supported_generation_methods", None)
                or []
            )
            if supported and not any("generatecontent" in str(s).lower() for s in supported):
                continue
            usable.append(short_name)

        if not usable:
            return []

        latest_flash = [c for c in usable if "latest" in c.lower() and "flash" in c.lower()]
        other_flash = [c for c in usable if "flash" in c.lower() and c not in latest_flash]
        others = [c for c in usable if c not in latest_flash and c not in other_flash]
        return latest_flash + other_flash + others

    @staticmethod
    def is_retryable_model_error(e: Exception) -> bool:
        msg = str(e)
        return any(code in msg for code in ("RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404"))

    def advance_model(self) -> bool:
        with self._lock:
            if self.model_index + 1 < len(self.models):
                self.model_index += 1
                self.model_name = self.models[self.model_index]
                logger.warning(f"[{self.label}] 모델 사용 불가로 전환 → {self.model_name}")
                return True
            return False

    def generate(self, contents, config=None, max_transient_retries: int = 2):
        switches_used = 0
        transient_left = max_transient_retries
        last_err = None
        while True:
            try:
                return self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            except Exception as e:
                last_err = e
                if self.is_retryable_model_error(e):
                    if switches_used >= self.MAX_MODEL_SWITCHES_PER_CALL:
                        logger.error(f"[{self.label}] 모델 전환 한도({self.MAX_MODEL_SWITCHES_PER_CALL}회) 초과, 포기")
                        raise last_err
                    if self.advance_model():
                        switches_used += 1
                        transient_left = max_transient_retries
                        continue
                    raise last_err
                transient_left -= 1
                if transient_left < 0:
                    raise last_err
                wait = 1.5 * (max_transient_retries - transient_left)
                logger.warning(f"[{self.label}] Gemini 호출 실패, {wait:.1f}초 후 재시도: {e}")
                time.sleep(wait)

    def create_chat(self, history=None, config=None):
        return self.client.chats.create(model=self.model_name, config=config, history=history or [])
