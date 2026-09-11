import io
import re
import time
import logging
import threading
from typing import Callable, Dict, Any, List, Union, Optional, Tuple
import telebot
import pandas as pd
from telebot.types import Message
from google.genai import types

from common import split_message, format_token_usage, log_token_usage, TelegramBotBase
from router import GeminiRouter
from store import ChatHistoryStore, KnowledgeStore

logger = logging.getLogger(__name__)

EMPTY_REPLY_FALLBACK = "응답이 비어 있습니다. 다시 한번 시도해 주세요."
THINKING_MESSAGE = "답변 준비중입니다..."
SAVING_MESSAGE = "자료 저장 중입니다..."
REMEMBER_TRIGGER = "저장해줘"
EXTRACTION_PROMPT = (
    "이 문서의 전체 내용을 최대한 원문 그대로 텍스트로 옮겨 적어줘. "
    "표가 있으면 구조를 유지하고, 요약하지 말고 전체 내용을 빠짐없이 옮겨 적어줘."
)
GROUP_DEBOUNCE_SECONDS = 3.0
MAX_AUTO_KB_MATCHES = 2
MAX_KB_CHARS_PER_DOC = 6000
HISTORY_LOAD_LIMIT = 8
SESSION_MAX_TURNS = 3           # 이 턴 수에 도달하면 계속 기억할지 사용자에게 확인 (안전장치, 토큰 조회 실패 시에도 동작)
SESSION_HARD_LIMIT_TURNS = 5    # 응답이 없어도 이 턴 수를 넘으면 강제로 정리 (안전장치)
SESSION_TOKEN_SOFT_LIMIT = 6000   # 직전 응답의 입력 토큰이 이 값을 넘으면 계속 기억할지 확인
SESSION_TOKEN_HARD_LIMIT = 12000  # 이 값을 넘으면 확인 없이 강제로 정리
SESSION_IDLE_TIMEOUT_SECONDS = 2 * 60 * 60
MAX_CACHED_SESSIONS = 200
MIN_LOCAL_PDF_TEXT_LENGTH = 100
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # Telegram Bot API가 이보다 큰 파일은 getFile로 받을 수 없음


class FileTooLargeError(Exception):
    pass

STOPWORDS = {
    "그리고", "그런데", "그래서", "하지만", "그러면", "저장해줘", "알려줘", "해줘",
    "것을", "것은", "인지", "입니다", "합니다", "있나요", "있어요", "얼마나", "무엇",
}

YES_KEYWORDS = {"응", "네", "예", "계속", "유지", "기억", "yes", "y", "ok", "오케이"}
NO_KEYWORDS = {"아니오", "아니요", "아니", "지워", "삭제", "초기화", "reset", "no", "n", "그만"}

CONTINUE_CONFIRM_TEXT = (
    "지금까지의 대화가 길어져서, 이 상태로 계속하면 질문마다 입력 토큰이 커집니다.\n"
    "이전 내용을 계속 기억해서 이어갈까요? '응' 또는 '아니오'로 답해주세요."
)
AUTO_RESET_NOTICE_TEXT = (
    "대화가 많이 길어져 토큰 절약을 위해 오래된 맥락을 자동으로 정리했습니다. "
    "이어서 질문해 주세요. (영구 참고자료는 유지됩니다)"
)

def extract_remember_name(caption: str, fallback_name: str) -> Tuple[Optional[str], bool]:
    caption = caption.strip()
    if not caption.startswith(REMEMBER_TRIGGER):
        return None, False
    rest = caption[len(REMEMBER_TRIGGER):].strip(" :-")
    if rest:
        return rest, False
    return fallback_name, True

class MemoryGeminiBot(TelegramBotBase):
    def __init__(self, name: str, token: str, router: GeminiRouter, store: ChatHistoryStore,
                 base_instruction: Union[str, Dict[str, str]], welcome_message: str,
                 quick_commands: Optional[Dict[str, str]] = None,
                 knowledge_store: Optional[KnowledgeStore] = None,
                 enable_token_usage: bool = False,
                 enable_session_confirmation: bool = False,
                 complexity_classifier: Optional[Callable[[str], str]] = None,
                 temperature: Optional[Union[float, Dict[str, float]]] = None,
                 max_output_tokens: Optional[Union[int, Dict[str, int]]] = None,
                 history_load_limit: int = HISTORY_LOAD_LIMIT,
                 session_max_turns: int = SESSION_MAX_TURNS,
                 session_hard_limit_turns: int = SESSION_HARD_LIMIT_TURNS,
                 session_token_soft_limit: int = SESSION_TOKEN_SOFT_LIMIT,
                 session_token_hard_limit: int = SESSION_TOKEN_HARD_LIMIT):
        self.name = name
        self.bot = telebot.TeleBot(token, threaded=False)
        self.router = router
        self.store = store
        self.base_instruction = base_instruction
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.welcome_message = welcome_message
        self.quick_commands = quick_commands or {}
        self.knowledge_store = knowledge_store
        self.enable_token_usage = enable_token_usage
        self.enable_session_confirmation = enable_session_confirmation
        self.complexity_classifier = complexity_classifier
        self.history_load_limit = history_load_limit
        self.session_max_turns = session_max_turns
        self.session_hard_limit_turns = session_hard_limit_turns
        self.session_token_soft_limit = session_token_soft_limit
        self.session_token_hard_limit = session_token_hard_limit
        self.user_sessions: Dict[int, Any] = {}
        self.session_last_used: Dict[int, float] = {}
        self.session_tier: Dict[int, str] = {}
        self.session_turn_count: Dict[int, int] = {}
        self.awaiting_confirm: Dict[int, bool] = {}
        self.pending_notice: Dict[int, str] = {}
        self.search_enabled: Dict[int, bool] = {}
        self.show_tokens: Dict[int, bool] = {}
        self._group_lock = threading.Lock()
        self._pending_groups: Dict[str, dict] = {}
        self._register_handlers()

    def _forget_session(self, chat_id: int):
        self.user_sessions.pop(chat_id, None)
        self.session_last_used.pop(chat_id, None)
        self.session_tier.pop(chat_id, None)
        self.session_turn_count.pop(chat_id, None)
        self.awaiting_confirm.pop(chat_id, None)
        self.pending_notice.pop(chat_id, None)

    def _tier_for(self, text: str) -> str:
        if not self.complexity_classifier:
            return "flash"
        try:
            return self.complexity_classifier(text)
        except Exception:
            return "flash"

    def _evict_stale_sessions(self):
        now = time.time()
        stale = [cid for cid, ts in self.session_last_used.items() if now - ts > SESSION_IDLE_TIMEOUT_SECONDS]
        for cid in stale:
            self._forget_session(cid)
            logger.info(f"[{self.name}] 장시간 미사용 세션 정리: Chat ID {cid}")

        if len(self.user_sessions) > MAX_CACHED_SESSIONS:
            oldest_first = sorted(self.session_last_used.items(), key=lambda kv: kv[1])
            excess = len(self.user_sessions) - MAX_CACHED_SESSIONS
            for cid, _ in oldest_first[:excess]:
                self._forget_session(cid)
                logger.info(f"[{self.name}] 세션 상한 초과로 정리: Chat ID {cid}")

    def _unique_fallback_name(self, chat_id: int, base_name: str) -> str:
        if not self.knowledge_store:
            return base_name
        existing = set(self.knowledge_store.list_names(chat_id))
        if base_name not in existing:
            return base_name
        i = 2
        while f"{base_name} ({i})" in existing:
            i += 1
        return f"{base_name} ({i})"

    @staticmethod
    def _resolve_by_tier(value, tier: str):
        """value가 등급별 dict({'flash': ..., 'pro': ...})면 해당 등급의 값을, 아니면 그대로 반환한다.
        base_instruction/temperature/max_output_tokens 모두 이 규칙을 공유한다."""
        if isinstance(value, dict):
            return value.get(tier) or value.get("flash") or next(iter(value.values()))
        return value

    def _build_config(self, chat_id: int, search_on: bool, tier: str = "flash") -> types.GenerateContentConfig:
        instruction = self._resolve_by_tier(self.base_instruction, tier)
        if self.knowledge_store:
            names = self.knowledge_store.list_names(chat_id)
            if names:
                name_list = ", ".join(names)
                instruction = (
                    f"{instruction}\n\n"
                    "[등록된 영구 참고자료 목록 — 아래 이름의 자료가 저장되어 있다. "
                    "사용자 질문이 이 자료들과 관련 있어 보이면, 관련된 부분만 발췌되어 함께 전달된다]\n"
                    f"{name_list}"
                )
        tools = [types.Tool(google_search=types.GoogleSearch())] if search_on else None
        config_kwargs = {"system_instruction": instruction, "tools": tools}
        temperature = self._resolve_by_tier(self.temperature, tier)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        max_output_tokens = self._resolve_by_tier(self.max_output_tokens, tier)
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        return types.GenerateContentConfig(**config_kwargs)

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.split(r"[\s\-_/().,?!\"'。、，:;]+", text)
        return list({t.lower() for t in tokens if len(t) >= 2 and t.lower() not in STOPWORDS})

    def _find_relevant_kb(self, chat_id: int, query: str) -> List[str]:
        if not self.knowledge_store:
            return []
        names = self.knowledge_store.list_names(chat_id)
        if not names:
            return []
        query_lower = query.lower()
        query_compact = query.replace(" ", "").lower()
        matched = []
        for name in names:
            name_compact = name.replace(" ", "").lower()
            if name_compact and name_compact in query_compact:
                matched.append(name)
                continue
            tokens = [t for t in re.split(r"[\s\-_/().,]+", name) if len(t) >= 2]
            if any(t.lower() in query_lower for t in tokens):
                matched.append(name)
        return matched[:MAX_AUTO_KB_MATCHES]

    def _build_prompt_with_kb(self, chat_id: int, user_text: str) -> Tuple[Union[str, list], List[str]]:
        matched_names = self._find_relevant_kb(chat_id, user_text)
        if not matched_names:
            return user_text, []
        keywords = self._extract_keywords(user_text)
        parts = []
        for name in matched_names:
            excerpt = self.knowledge_store.get_relevant_excerpt(chat_id, name, keywords, MAX_KB_CHARS_PER_DOC)
            if excerpt:
                parts.append(f"[참고자료: {name}]\n{excerpt}")
        parts.append(user_text)
        logger.info(f"[{self.name}] 참고자료 자동 매칭: {matched_names} (문서당 최대 {MAX_KB_CHARS_PER_DOC:,}자 발췌)")
        return parts, matched_names

    def _thinking_text_for(self, matched_names: List[str]) -> str:
        if not matched_names:
            return THINKING_MESSAGE
        names_str = "', '".join(matched_names)
        return f"저장된 자료('{names_str}')를 참고해서 답변 준비중입니다..."

    def _history_to_genai_format(self, rows: List[Dict[str, str]]) -> List[types.Content]:
        return [types.Content(role=r["role"], parts=[types.Part(text=r["content"])]) for r in rows]

    def _get_chat_session(self, chat_id: int, tier: str = "flash"):
        self.session_last_used[chat_id] = time.time()
        if chat_id in self.user_sessions and self.session_tier.get(chat_id) != tier:
            logger.info(
                f"[{self.name}] 등급 전환({self.session_tier.get(chat_id)} → {tier})으로 세션 재생성: Chat ID {chat_id}"
            )
            self._forget_session(chat_id)
        if chat_id not in self.user_sessions:
            self._evict_stale_sessions()
            history = self._history_to_genai_format(self.store.load_history(chat_id, limit=self.history_load_limit))
            search_on = self.search_enabled.get(chat_id, False)
            config = self._build_config(chat_id, search_on, tier)
            self.user_sessions[chat_id] = self.router.create_chat(history=history, config=config, tier=tier)
            self.session_tier[chat_id] = tier
            self.session_last_used[chat_id] = time.time()
            self.session_turn_count[chat_id] = 0
            logger.info(
                f"[{self.name}] 세션 생성/복원: Chat ID {chat_id} "
                f"(기록 {len(history)}건, 검색={'ON' if search_on else 'OFF'}, 등급={tier}, "
                f"모델={self.router.current_model(tier)}, 캐시된 세션 수={len(self.user_sessions)})"
            )
        return self.user_sessions[chat_id]

    def _register_turn(self, chat_id: int, response=None):
        count = self.session_turn_count.get(chat_id, 0) + 1
        self.session_turn_count[chat_id] = count

        prompt_tokens = None
        usage = getattr(response, "usage_metadata", None) if response is not None else None
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)

        hit_hard_limit = count >= self.session_hard_limit_turns or (
            prompt_tokens is not None and prompt_tokens >= self.session_token_hard_limit
        )
        hit_soft_limit = count >= self.session_max_turns or (
            prompt_tokens is not None and prompt_tokens >= self.session_token_soft_limit
        )

        if hit_hard_limit:
            logger.info(
                f"[{self.name}] 세션 강제 한도 도달({count}턴, 입력 토큰 {prompt_tokens}), "
                f"자동으로 세션 재구성 (Chat ID {chat_id})"
            )
            self._forget_session(chat_id)
            if self.enable_session_confirmation:
                self.pending_notice[chat_id] = AUTO_RESET_NOTICE_TEXT
            return

        if self.enable_session_confirmation and hit_soft_limit and not self.awaiting_confirm.get(chat_id):
            self.awaiting_confirm[chat_id] = True
            self.pending_notice[chat_id] = CONTINUE_CONFIRM_TEXT
            logger.info(
                f"[{self.name}] 세션 누적 컨텍스트 한도 도달({count}턴, 입력 토큰 {prompt_tokens}), "
                f"사용자에게 유지 여부 확인 (Chat ID {chat_id})"
            )

    def _handle_continue_confirmation(self, chat_id: int, text: str) -> bool:
        normalized = text.strip().lower()
        if any(k in normalized for k in NO_KEYWORDS):
            self._forget_session(chat_id)
            self.store.clear(chat_id)
            self.bot.send_message(chat_id, "이전 대화 내용을 정리했습니다. 새로운 질문을 입력해 주세요. (영구 참고자료는 유지됩니다)")
            return True
        if any(k in normalized for k in YES_KEYWORDS):
            self.awaiting_confirm.pop(chat_id, None)
            self.session_turn_count[chat_id] = 0
            self.bot.send_message(chat_id, "알겠습니다. 이전 내용을 계속 이어서 기억하겠습니다.")
            return True
        # 명확한 응답이 아니면 확인은 종료하고, 방금 메시지는 새 질문으로 처리(기본은 '계속 유지')
        self.awaiting_confirm.pop(chat_id, None)
        return False

    def _send_with_retry(self, chat_id: int, content: Union[str, list], tier: str = "flash",
                          max_transient_retries: int = 2):
        switches_used = 0
        transient_left = max_transient_retries
        last_err = None
        current_tier = tier
        while True:
            chat_session = self._get_chat_session(chat_id, current_tier)
            try:
                response = chat_session.send_message(content)
                log_token_usage(self.name, response)
                self._register_turn(chat_id, response)
                return response
            except Exception as e:
                last_err = e
                if self.router.is_retryable_model_error(e):
                    if switches_used >= self.router.MAX_MODEL_SWITCHES_PER_CALL:
                        logger.error(f"[{self.name}] 모델 전환 한도 초과, 포기")
                        # 이 세션은 실패가 확인된 모델에 그대로 묶여 있으므로, 다음 메시지에서
                        # 새 모델로 세션을 재생성하도록 캐시를 비운다.
                        self._forget_session(chat_id)
                        raise last_err
                    if self.router.advance_model(current_tier):
                        switches_used += 1
                        self._forget_session(chat_id)
                        transient_left = max_transient_retries
                        continue
                    if current_tier != "flash":
                        # 상위 등급이 전부 소진되면 가용성을 위해 flash로 강등해 계속 시도한다.
                        current_tier = "flash"
                        switches_used += 1
                        self._forget_session(chat_id)
                        transient_left = max_transient_retries
                        continue
                    # 더 이상 전환할 모델이 없는 경우에도 동일하게 세션을 비워,
                    # 죽은 모델에 계속 묶이지 않게 한다.
                    self._forget_session(chat_id)
                    raise last_err
                transient_left -= 1
                if transient_left < 0:
                    raise last_err
                wait = 1.5 * (max_transient_retries - transient_left)
                logger.warning(f"[{self.name}] 응답 실패, {wait:.1f}초 후 재시도: {e}")
                time.sleep(wait)

    def _reply(self, chat_id: int, text: str, edit_message_id: Optional[int] = None):
        chunks = split_message(text)
        for i, chunk in enumerate(chunks):
            if i == 0 and edit_message_id is not None:
                try:
                    self.bot.edit_message_text(chunk, chat_id=chat_id, message_id=edit_message_id, parse_mode='Markdown')
                    continue
                except Exception:
                    try:
                        self.bot.edit_message_text(chunk, chat_id=chat_id, message_id=edit_message_id)
                        continue
                    except Exception:
                        pass
            try:
                self.bot.send_message(chat_id, chunk, parse_mode='Markdown')
            except Exception:
                self.bot.send_message(chat_id, chunk)

    def _show_error(self, chat_id: int, text: str, edit_message_id: Optional[int] = None):
        if edit_message_id is not None:
            try:
                self.bot.edit_message_text(text, chat_id=chat_id, message_id=edit_message_id)
                return
            except Exception:
                pass
        self.bot.send_message(chat_id, text)

    def _save_history_safely(self, chat_id: int, user_input: str, reply_text: str):
        try:
            self.store.append(chat_id, "user", user_input)
            self.store.append(chat_id, "model", reply_text)
        except Exception as e:
            logger.warning(f"[{self.name}] 대화 기록 저장 실패(응답은 정상 전달됨): {e}")

    def _excel_to_text(self, file_bytes: bytes) -> str:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine="openpyxl")
        parts = []
        for sheet_name, df in sheets.items():
            parts.append(f"[시트: {sheet_name}]\n{df.to_csv(index=False)}")
        return "\n\n".join(parts)

    def _pdf_to_text_local(self, file_bytes: bytes) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            parts = []
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    parts.append(f"[페이지 {i + 1}]\n{text}")
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"[{self.name}] 로컬 PDF 텍스트 추출 실패: {e}")
            return ""

    def _download_file_payload(self, message: Message):
        if message.content_type == 'document':
            file_name = message.document.file_name or "문서"
            mime_type = message.document.mime_type or "application/octet-stream"
            file_size = message.document.file_size
            file_id = message.document.file_id
        else:
            file_name = "사진"
            mime_type = "image/jpeg"
            file_size = message.photo[-1].file_size
            file_id = message.photo[-1].file_id

        if file_size and file_size > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"{file_name} ({file_size / 1024 / 1024:.1f}MB)이 최대 허용 크기 "
                f"{MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB를 초과합니다."
            )

        file_info = self.bot.get_file(file_id)
        file_bytes = self.bot.download_file(file_info.file_path)
        return file_name, file_bytes, mime_type

    def _extract_kb_text(self, file_name: str, file_bytes: bytes, mime_type: str) -> str:
        if file_name.lower().endswith((".xlsx", ".xls")):
            return self._excel_to_text(file_bytes)

        if file_name.lower().endswith(".pdf") or mime_type == "application/pdf":
            local_text = self._pdf_to_text_local(file_bytes)
            if len(local_text.strip()) >= MIN_LOCAL_PDF_TEXT_LENGTH:
                logger.info(f"[{self.name}] PDF 텍스트를 로컬에서 무료로 추출함 (Gemini 미사용)")
                return local_text
            logger.info(f"[{self.name}] PDF에서 추출 가능한 텍스트가 부족함(스캔본 추정), Gemini로 대체 추출")

        response = self.router.generate(
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), EXTRACTION_PROMPT]
        )
        log_token_usage(self.name, response)
        return response.text or "(추출된 내용 없음)"

    def _process_and_reply(self, chat_id: int, history_label: str, model_prompt: Union[str, list],
                            thinking_text: str = THINKING_MESSAGE, tier: str = "flash"):
        self.bot.send_chat_action(chat_id, 'typing')
        placeholder = self.bot.send_message(chat_id, thinking_text)
        try:
            response = self._send_with_retry(chat_id, model_prompt, tier=tier)
            reply_text = response.text or EMPTY_REPLY_FALLBACK
            display_text = reply_text
            if self.enable_token_usage and self.show_tokens.get(chat_id, True):
                display_text += format_token_usage(response)
            self._reply(chat_id, display_text, edit_message_id=placeholder.message_id)
            self._save_history_safely(chat_id, history_label, reply_text)
            notice = self.pending_notice.pop(chat_id, None)
            if notice:
                self.bot.send_message(chat_id, notice)
        except Exception as e:
            logger.error(f"[{self.name}] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
            self._show_error(
                chat_id,
                "오류가 발생했습니다. 잠시 후 다시 시도하거나 /reset을 입력해 주세요.",
                edit_message_id=placeholder.message_id
            )

    def _make_quick_command_handler(self, cmd_name: str, template: str):
        def handler(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self._ai_cooldown_ok(chat_id):
                return
            extra = message.text.partition(' ')[2].strip()
            base_text = template + (f"\n\n[추가 참고 사항]: {extra}" if extra else "")
            prompt, matched_names = self._build_prompt_with_kb(chat_id, base_text)
            history_label = f"/{cmd_name} {extra}".strip()
            # 전문 분야 단축 명령어는 정확도가 중요하므로, 등급 자동분기가 켜진 봇에서는 항상 pro로 처리한다.
            tier = "pro" if self.complexity_classifier else "flash"
            self._process_and_reply(chat_id, history_label, prompt, thinking_text=self._thinking_text_for(matched_names), tier=tier)
        return handler

    # ---------------- 앨범(여러 파일 묶음) 처리 ----------------

    def _buffer_group_message(self, message: Message):
        group_id = message.media_group_id
        with self._group_lock:
            group = self._pending_groups.get(group_id)
            if group is None:
                group = {"chat_id": message.chat.id, "items": [], "caption": None, "timer": None}
                self._pending_groups[group_id] = group
            group["items"].append(message)
            if message.caption:
                group["caption"] = message.caption
            if group["timer"]:
                group["timer"].cancel()
            timer = threading.Timer(GROUP_DEBOUNCE_SECONDS, self._process_group, args=(group_id,))
            group["timer"] = timer
            timer.start()

    def _process_group(self, group_id: str):
        with self._group_lock:
            group = self._pending_groups.pop(group_id, None)
        if not group:
            return
        chat_id = group["chat_id"]
        if not self._guard(chat_id):
            return

        caption = (group["caption"] or "").strip()
        items = group["items"]
        remember_base = None
        if caption == REMEMBER_TRIGGER:
            remember_base = None
        elif caption.startswith(REMEMBER_TRIGGER):
            remember_base = caption[len(REMEMBER_TRIGGER):].strip(" :-") or None
            if remember_base is None:
                remember_base = ""

        if caption.startswith(REMEMBER_TRIGGER):
            if not self.knowledge_store:
                self.bot.send_message(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.")
                return
            notice = self.bot.send_message(chat_id, f"파일 {len(items)}개 저장 중입니다...")
            saved, failed = [], []
            for msg in items:
                try:
                    file_name, file_bytes, mime_type = self._download_file_payload(msg)
                    base_kb_name = f"{remember_base} - {file_name}" if remember_base else file_name
                    kb_name = self._unique_fallback_name(chat_id, base_kb_name)
                    text = self._extract_kb_text(file_name, file_bytes, mime_type)
                    self.knowledge_store.add(chat_id, kb_name, text)
                    saved.append(kb_name)
                except FileTooLargeError as e:
                    logger.warning(f"[{self.name}] 그룹 파일 저장 건너뜀(용량 초과): {e}")
                    failed.append(str(e))
                except Exception as e:
                    logger.error(f"[{self.name}] 그룹 파일 저장 실패: {e}", exc_info=True)
                    failed.append(getattr(msg.document, "file_name", "알 수 없는 파일") if msg.content_type == 'document' else "사진")
            self._forget_session(chat_id)
            result = "저장 완료:\n" + "\n".join(f"- {n}" for n in saved)
            if failed:
                result += "\n\n저장 실패:\n" + "\n".join(f"- {n}" for n in failed)
            try:
                self.bot.edit_message_text(result, chat_id=chat_id, message_id=notice.message_id)
            except Exception:
                self.bot.send_message(chat_id, result)
        else:
            self.bot.send_message(chat_id, f"파일 {len(items)}개를 받았습니다. 여러 파일 동시 분석은 지원하지 않아 첫 번째 파일만 분석합니다.\n"
                                            f"전체를 저장하려면 캡션에 '{REMEMBER_TRIGGER}'를 붙여 다시 보내주세요.")
            self._handle_single_file(items[0])

    # ---------------- 단일 파일 처리 ----------------

    def _handle_single_file(self, message: Message):
        chat_id = message.chat.id
        if not self._guard(chat_id):
            return

        caption = message.caption or ""
        try:
            file_name, file_bytes, mime_type = self._download_file_payload(message)
        except FileTooLargeError as e:
            self.bot.send_message(chat_id, f"파일이 너무 큽니다: {e}")
            return
        except Exception as e:
            logger.error(f"[{self.name}] 파일 다운로드 실패 (Chat ID: {chat_id}): {e}", exc_info=True)
            self.bot.send_message(chat_id, "파일을 받는 중 오류가 발생했습니다.")
            return

        remember_name, is_fallback = extract_remember_name(caption, file_name)

        self.bot.send_chat_action(chat_id, 'typing')
        placeholder = self.bot.send_message(chat_id, SAVING_MESSAGE if remember_name else THINKING_MESSAGE)

        try:
            if remember_name:
                if not self.knowledge_store:
                    self._show_error(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.", edit_message_id=placeholder.message_id)
                    return
                final_name = self._unique_fallback_name(chat_id, remember_name) if is_fallback else remember_name
                text = self._extract_kb_text(file_name, file_bytes, mime_type)
                self.knowledge_store.add(chat_id, final_name, text)
                self._forget_session(chat_id)
                self._reply(chat_id, f"'{final_name}' 자료로 저장했습니다. 앞으로 관련 질문에 자동으로 참고합니다.", edit_message_id=placeholder.message_id)
                return

            if file_name.lower().endswith((".xlsx", ".xls")):
                content_parts = [self._excel_to_text(file_bytes), caption or "이 파일의 내용을 분석하고 핵심을 요약해줘."]
            else:
                content_parts = [types.Part.from_bytes(data=file_bytes, mime_type=mime_type), caption or "이 파일의 내용을 분석하고 핵심을 요약해줘."]

            # 첨부파일 분석은 정확도가 중요하므로, 등급 자동분기가 켜진 봇에서는 항상 pro로 처리한다.
            tier = "pro" if self.complexity_classifier else "flash"
            response = self._send_with_retry(chat_id, content_parts, tier=tier)
            reply_text = response.text or EMPTY_REPLY_FALLBACK
            display_text = reply_text
            if self.enable_token_usage and self.show_tokens.get(chat_id, True):
                display_text += format_token_usage(response)
            self._reply(chat_id, display_text, edit_message_id=placeholder.message_id)
            self._save_history_safely(chat_id, f"[파일 첨부] {caption or '분석 요청'}", reply_text)
            notice = self.pending_notice.pop(chat_id, None)
            if notice:
                self.bot.send_message(chat_id, notice)
        except Exception as e:
            logger.error(f"[{self.name}] 파일 처리 예외 (Chat ID: {chat_id}): {e}", exc_info=True)
            self._show_error(
                chat_id,
                "파일 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                edit_message_id=placeholder.message_id
            )

    def _register_handlers(self):
        self._register_common_handlers()

        @self.bot.message_handler(commands=['model'])
        def handle_model(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if self.complexity_classifier:
                self.bot.send_message(
                    chat_id,
                    f"Flash 모델(간단한 질문): {self.router.current_model('flash')}\n"
                    f"Pro 모델(전문/복잡한 질문): {self.router.current_model('pro')}\n"
                    "질문 난이도에 따라 자동으로 등급이 선택됩니다."
                )
            else:
                self.bot.send_message(chat_id, f"현재 사용 중인 모델: {self.router.current_model('flash')}")

        @self.bot.message_handler(commands=['search'])
        def handle_search_toggle(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            parts = message.text.split()
            if len(parts) < 2 or parts[1].lower() not in ('on', 'off'):
                current = "켜짐" if self.search_enabled.get(chat_id, False) else "꺼짐"
                self.bot.send_message(chat_id, f"현재 검색 기능: {current}\n사용법: /search on 또는 /search off")
                return
            enable = parts[1].lower() == 'on'
            self.search_enabled[chat_id] = enable
            self._forget_session(chat_id)
            self.bot.send_message(chat_id, f"검색 기능을 {'켰습니다' if enable else '껐습니다'}.")

        if self.enable_token_usage:
            @self.bot.message_handler(commands=['tokens'])
            def handle_tokens_toggle(message: Message):
                chat_id = message.chat.id
                if not self._guard(chat_id):
                    return
                parts = message.text.split()
                if len(parts) < 2 or parts[1].lower() not in ('on', 'off'):
                    current = "켜짐" if self.show_tokens.get(chat_id, True) else "꺼짐"
                    self.bot.send_message(chat_id, f"현재 토큰 사용량 표시: {current}\n사용법: /tokens on 또는 /tokens off")
                    return
                enable = parts[1].lower() == 'on'
                self.show_tokens[chat_id] = enable
                self.bot.send_message(chat_id, f"토큰 사용량 표시를 {'켰습니다' if enable else '껐습니다'}.")

        @self.bot.message_handler(commands=['kb'])
        def handle_kb_list(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self.knowledge_store:
                self.bot.send_message(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.")
                return
            names = self.knowledge_store.list_names(chat_id)
            if not names:
                self.bot.send_message(chat_id, "저장된 참고자료가 없습니다.\n파일 보낼 때 캡션에 '저장해줘' 또는 '저장해줘 문서이름'이라고 적어서 등록하세요.")
                return
            listing = "\n".join(f"- {n}" for n in names)
            self.bot.send_message(chat_id, f"저장된 참고자료 목록:\n{listing}\n\n질문에 이 이름이나 관련 키워드가 들어가면 자동으로 불러와서 참고합니다.")

        @self.bot.message_handler(commands=['forget'])
        def handle_forget(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self.knowledge_store:
                self.bot.send_message(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.")
                return
            name = message.text.partition(' ')[2].strip()
            if not name:
                self.bot.send_message(chat_id, "사용법: /forget 문서이름")
                return
            if self.knowledge_store.remove(chat_id, name):
                self._forget_session(chat_id)
                self.bot.send_message(chat_id, f"'{name}' 자료를 삭제했습니다.")
            else:
                self.bot.send_message(chat_id, f"'{name}'이라는 이름의 저장된 자료를 찾지 못했습니다. /kb로 목록을 확인하세요.")

        @self.bot.message_handler(commands=['export'])
        def handle_export(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self.knowledge_store:
                self.bot.send_message(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.")
                return
            name = message.text.partition(' ')[2].strip()
            if not name:
                self.bot.send_message(chat_id, "사용법: /export 문서이름 (목록은 /kb)")
                return
            content = self.knowledge_store.get(chat_id, name)
            if content is None:
                self.bot.send_message(chat_id, f"'{name}'이라는 이름의 저장된 자료를 찾지 못했습니다. /kb로 목록을 확인하세요.")
                return
            buffer = io.BytesIO(content.encode('utf-8'))
            buffer.name = f"{name}.txt"
            self.bot.send_document(
                chat_id, buffer,
                caption=f"'{name}' 저장된 내용입니다. (원본 엑셀/PDF가 아니라, 봇이 답변에 쓰는 텍스트 추출본입니다)"
            )

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            quick_list = "\n".join(f"/{c} - {t[:28]}..." for c, t in self.quick_commands.items())
            quick_section = f"\n\n[전문 분야 단축 명령어]\n{quick_list}" if quick_list else ""
            kb_section = (
                "\n\n[영구 참고자료]\n"
                "파일 보낼 때 캡션에 '저장해줘' 또는 '저장해줘 문서이름'이라고 적으면 영구 저장됩니다 (reset해도 안 사라짐).\n"
                "여러 파일을 한꺼번에 보낼 때도, 그중 아무 파일에나 캡션으로 '저장해줘'를 붙이면 전부 저장됩니다.\n"
                "저장된 자료는 평소엔 이름만 기억하고 있다가, 질문에 관련 이름/키워드가 나오면 그때만 "
                "관련된 부분만 발췌해서 참고합니다 (문서 전체를 매번 불러오지 않아 비용이 절감됩니다).\n"
                "/kb - 저장된 자료 목록 확인\n"
                "/export 문서이름 - 저장된 자료를 텍스트 파일로 다운로드 (원본 파일이 아닌 추출된 텍스트)\n"
                "/forget 문서이름 - 저장된 자료 삭제"
                if self.knowledge_store else ""
            )
            tokens_line = "/tokens on|off - 답변마다 토큰 사용량 표시 켜기/끄기 (기본 켜짐)\n" if self.enable_token_usage else ""
            tier_line = (
                "/model - 질문 난이도에 따라 자동 선택되는 모델 확인 "
                "(계산·법령·재무 등 전문 질문은 Pro, 일상 질문은 Flash)\n"
                if self.complexity_classifier else "/model - 현재 사용 모델 확인\n"
            )
            session_note = (
                "\n대화가 길어지면 계속 이어갈지 확인 메시지가 뜹니다. "
                "'응'이면 계속, '아니오'면 대화 기록을 정리합니다."
                if self.enable_session_confirmation else
                "\n대화가 너무 길어지면 별도 확인 없이 자동으로 오래된 맥락을 정리합니다."
            )
            self.bot.send_message(
                chat_id,
                f"{self.welcome_message}\n\n"
                "/search on|off - 최신 정보 검색 기능 켜기/끄기 (기본 꺼짐)\n"
                f"{tokens_line}"
                "/reset - 대화 기록 초기화\n"
                f"{tier_line}"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/usage - 오늘/최근 7일 토큰 사용량 확인 (전체 봇 합산)\n"
                "/myid - 내 chat_id 확인\n"
                "/help - 이 도움말 보기\n"
                f"{session_note}"
                f"{quick_section}{kb_section}"
            )

        @self.bot.message_handler(commands=['start', 'reset'])
        def handle_commands(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            command = message.text.split()[0].lower()
            if command == '/start':
                self.bot.send_message(chat_id, self.welcome_message, parse_mode='Markdown')
            elif command == '/reset':
                self._forget_session(chat_id)
                self.store.clear(chat_id)
                self.bot.send_message(chat_id, "대화 기록이 초기화되었습니다. (영구 참고자료는 유지됩니다)")

        for cmd_name, template in self.quick_commands.items():
            self.bot.message_handler(commands=[cmd_name])(self._make_quick_command_handler(cmd_name, template))

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if self.awaiting_confirm.get(chat_id) and self._handle_continue_confirmation(chat_id, message.text):
                return
            if not self._ai_cooldown_ok(chat_id):
                return  # 너무 빠른 연속 요청은 조용히 무시(실수 폭주 방지), 매번 경고하면 더 시끄러움
            prompt, matched_names = self._build_prompt_with_kb(chat_id, message.text)
            tier = self._tier_for(message.text)
            self._process_and_reply(chat_id, message.text, prompt, thinking_text=self._thinking_text_for(matched_names), tier=tier)

        @self.bot.message_handler(content_types=['document', 'photo'])
        def handle_file(message: Message):
            if message.media_group_id:
                self._buffer_group_message(message)
                return
            self._handle_single_file(message)

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
