import os
import time
import logging
import threading
from typing import List, Tuple
from google import genai

logger = logging.getLogger(__name__)


def resolve_api_key(specific_env: str) -> str:
    key = os.environ.get(specific_env) or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(f"{specific_env} 또는 GEMINI_API_KEY 환경변수가 필요합니다.")
    return key


class GeminiRouter:
    FALLBACK_FLASH_MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    FALLBACK_PRO_MODELS = [
        "gemini-3-pro-preview",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
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
        flash, pro = self._discover_models()
        self.flash_models = flash or list(self.FALLBACK_FLASH_MODELS)
        self.pro_models = pro or list(self.FALLBACK_PRO_MODELS)
        self.flash_index = 0
        self.pro_index = 0
        logger.info(
            f"[{self.label}] Flash 모델 {len(self.flash_models)}개, Pro 모델 {len(self.pro_models)}개 확인. "
            f"Flash 1순위: {self.flash_models[0]}, Pro 1순위: {self.pro_models[0] if self.pro_models else '없음'}"
        )

    def _discover_models(self) -> Tuple[List[str], List[str]]:
        try:
            raw_models = list(self.client.models.list())
        except Exception as e:
            logger.warning(f"[{self.label}] 모델 목록 조회 실패, 기본 후보 목록 사용: {e}")
            return [], []

        flash, pro, other = [], [], []
        for m in raw_models:
            name = getattr(m, "name", None)
            if not name:
                continue
            short_name = name.split("/")[-1]
            low = short_name.lower()
            if any(k in low for k in self.EXCLUDE_KEYWORDS):
                continue
            supported = (
                getattr(m, "supported_actions", None)
                or getattr(m, "supported_generation_methods", None)
                or []
            )
            if supported and not any("generatecontent" in str(s).lower() for s in supported):
                continue
            if "flash" in low:
                flash.append(short_name)
            elif "pro" in low:
                pro.append(short_name)
            else:
                other.append(short_name)

        # "-latest"류 별칭은 실제로 어떤 모델이 응답했는지 사용자가 알 수 없으므로,
        # 구체적인 버전이 박힌 모델명을 우선하고 별칭은 후순위 대체용으로만 둔다.
        def order(names: List[str]) -> List[str]:
            latest = [c for c in names if "latest" in c.lower()]
            rest = [c for c in names if c not in latest]
            return rest + latest

        flash = order(flash)
        # 분류 불가능한 모델(other)은 예상 밖 신모델 대비용으로 pro 등급 맨 뒤에 붙여둔다.
        pro = order(pro) + order(other)
        return flash, pro

    def _tier_list(self, tier: str) -> List[str]:
        return self.pro_models if tier == "pro" else self.flash_models

    def current_model(self, tier: str = "flash") -> str:
        lst = self._tier_list(tier)
        idx = self.pro_index if tier == "pro" else self.flash_index
        if lst:
            return lst[min(idx, len(lst) - 1)]
        other = self.flash_models if tier == "pro" else self.pro_models
        if other:
            return other[0]
        raise ValueError(f"[{self.label}] 사용 가능한 모델이 없습니다.")

    @property
    def model_name(self) -> str:
        return self.current_model("flash")

    @staticmethod
    def is_retryable_model_error(e: Exception) -> bool:
        msg = str(e)
        return any(code in msg for code in ("RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404"))

    def advance_model(self, tier: str = "flash") -> bool:
        """같은 등급 안에서 다음 모델로 전환. 등급 안에 더 이상 없으면 False."""
        with self._lock:
            lst = self._tier_list(tier)
            idx = self.pro_index if tier == "pro" else self.flash_index
            if idx + 1 >= len(lst):
                return False
            idx += 1
            if tier == "pro":
                self.pro_index = idx
            else:
                self.flash_index = idx
            logger.warning(f"[{self.label}] {tier} 등급 모델 전환 → {lst[idx]}")
        return True

    def generate(self, contents, config=None, tier: str = "flash", max_transient_retries: int = 2):
        switches_used = 0
        transient_left = max_transient_retries
        last_err = None
        current_tier = tier
        while True:
            model_name = self.current_model(current_tier)
            try:
                return self.client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
            except Exception as e:
                last_err = e
                if self.is_retryable_model_error(e):
                    if switches_used >= self.MAX_MODEL_SWITCHES_PER_CALL:
                        logger.error(f"[{self.label}] 모델 전환 한도({self.MAX_MODEL_SWITCHES_PER_CALL}회) 초과, 포기")
                        raise last_err
                    if self.advance_model(current_tier):
                        switches_used += 1
                        transient_left = max_transient_retries
                        continue
                    if current_tier != "flash":
                        # 상위 등급이 전부 소진되면 가용성을 위해 flash로 강등해 계속 시도한다.
                        current_tier = "flash"
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

    def create_chat(self, history=None, config=None, tier: str = "flash"):
        return self.client.chats.create(model=self.current_model(tier), config=config, history=history or [])
