import io
import time
import logging
from typing import Dict, Any, List, Union, Optional
import telebot
import pandas as pd
from telebot.types import Message
from google.genai import types

from common import is_allowed, split_message, get_uptime_str
from router import GeminiRouter
from store import ChatHistoryStore, KnowledgeStore

logger = logging.getLogger(__name__)

EMPTY_REPLY_FALLBACK = "⚠️ 응답이 비어 있습니다. 다시 한번 시도해 주세요."
THINKING_MESSAGE = "🤔 답변 준비중입니다..."
SAVING_MESSAGE = "📚 자료 저장 중입니다..."
REMEMBER_TRIGGER = "저장해줘"
EXTRACTION_PROMPT = (
    "이 문서의 전체 내용을 최대한 원문 그대로 텍스트로 옮겨 적어줘. "
    "표가 있으면 구조를 유지하고, 요약하지 말고 전체 내용을 빠짐없이 옮겨 적어줘."
)


def extract_remember_name(caption: str, fallback_name: str) -> Optional[str]:
    caption = caption.strip()
    if not caption.startswith(REMEMBER_TRIGGER):
        return None
    rest = caption[len(REMEMBER_TRIGGER):].strip(" :-")
    return rest or fallback_name


class MemoryGeminiBot:
    def __init__(self, name: str, token: str, router: GeminiRouter, store: ChatHistoryStore,
                 base_instruction: str, welcome_message: str,
                 quick_commands: Optional[Dict[str, str]] = None,
                 knowledge_store: Optional[KnowledgeStore] = None):
        self.name = name
        self.bot = telebot.TeleBot(token)
        self.router = router
        self.store = store
        self.base_instruction = base_instruction
        self.welcome_message = welcome_message
        self.quick_commands = quick_commands or {}
        self.knowledge_store = knowledge_store
        self.user_sessions: Dict[int, Any] = {}
        self.search_enabled: Dict[int, bool] = {}
        self._register_handlers()

    def _build_config(self, chat_id: int, search_on: bool) -> types.GenerateContentConfig:
        instruction = self.base_instruction
        if self.knowledge_store:
            kb_text = self.knowledge_store.get_all_text(chat_id)
            if kb_text:
                instruction = (
                    f"{instruction}\n\n"
                    "[영구 참고자료 — 사용자가 등록해둔 자료다. 관련 질문엔 항상 우선 참고하라]\n"
                    f"{kb_text}"
                )
        tools = [types.Tool(google_search=types.GoogleSearch())] if search_on else None
        return types.GenerateContentConfig(system_instruction=instruction, tools=tools)

    def _history_to_genai_format(self, rows: List[Dict[str, str]]) -> List[types.Content]:
        return [types.Content(role=r["role"], parts=[types.Part(text=r["content"])]) for r in rows]

    def _get_chat_session(self, chat_id: int):
        if chat_id not in self.user_sessions:
            history = self._history_to_genai_format(self.store.load_history(chat_id))
            search_on = self.search_enabled.get(chat_id, False)
            config = self._build_config(chat_id, search_on)
            self.user_sessions[chat_id] = self.router.create_chat(history=history, config=config)
            logger.info(
                f"[{self.name}] 세션 생성/복원: Chat ID {chat_id} "
                f"(기록 {len(history)}건, 검색={'ON' if search_on else 'OFF'}, 모델={self.router.model_name})"
            )
        return self.user_sessions[chat_id]

    def _send_with_retry(self, chat_id: int, content: Union[str, list], max_transient_retries: int = 2):
        switches_used = 0
        transient_left = max_transient_retries
        last_err = None
        while True:
            chat_session = self._get_chat_session(chat_id)
            try:
                return chat_session.send_message(content)
            except Exception as e:
                last_err = e
                if self.router.is_retryable_model_error(e):
                    if switches_used >= self.router.MAX_MODEL_SWITCHES_PER_CALL:
                        logger.error(f"[{self.name}] 모델 전환 한도 초과, 포기")
                        raise last_err
                    if self.router.advance_model():
                        switches_used += 1
                        self.user_sessions.pop(chat_id, None)
                        transient_left = max_transient_retries
                        continue
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

    def _process_and_reply(self, chat_id: int, history_label: str, model_prompt: Union[str, list]):
        self.bot.send_chat_action(chat_id, 'typing')
        placeholder = self.bot.send_message(chat_id, THINKING_MESSAGE)
        try:
            response = self._send_with_retry(chat_id, model_prompt)
            reply_text = response.text or EMPTY_REPLY_FALLBACK
            self._reply(chat_id, reply_text, edit_message_id=placeholder.message_id)
            self._save_history_safely(chat_id, history_label, reply_text)
        except Exception as e:
            logger.error(f"[{self.name}] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
            self._show_error(
                chat_id,
                "⚠️ 오류가 발생했습니다. 잠시 후 다시 시도하거나 /reset을 입력해 주세요.",
                edit_message_id=placeholder.message_id
            )

    def _make_quick_command_handler(self, cmd_name: str, template: str):
        def handler(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            extra = message.text.partition(' ')[2].strip()
            prompt = template + (f"\n\n[추가 참고 사항]: {extra}" if extra else "")
            history_label = f"/{cmd_name} {extra}".strip()
            self._process_and_reply(chat_id, history_label, prompt)
        return handler

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            self.bot.send_message(message.chat.id, f"🆔 chat_id: `{message.chat.id}`", parse_mode='Markdown')

        @self.bot.message_handler(commands=['uptime'])
        def handle_uptime(message: Message):
            self.bot.send_message(message.chat.id, f"⏱ 서버 연속 가동 시간: {get_uptime_str()}")

        @self.bot.message_handler(commands=['model'])
        def handle_model(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"🧠 현재 사용 중인 모델: `{self.router.model_name}`", parse_mode='Markdown')

        @self.bot.message_handler(commands=['search'])
        def handle_search_toggle(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            parts = message.text.split()
            if len(parts) < 2 or parts[1].lower() not in ('on', 'off'):
                current = "켜짐" if self.search_enabled.get(chat_id, False) else "꺼짐"
                self.bot.send_message(chat_id, f"현재 검색 기능: {current}\n사용법: /search on 또는 /search off")
                return
            enable = parts[1].lower() == 'on'
            self.search_enabled[chat_id] = enable
            self.user_sessions.pop(chat_id, None)
            self.bot.send_message(chat_id, f"🔍 검색 기능을 {'켰습니다' if enable else '껐습니다'}.")

        @self.bot.message_handler(commands=['kb'])
        def handle_kb_list(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            if not self.knowledge_store:
                self.bot.send_message(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.")
                return
            names = self.knowledge_store.list_names(chat_id)
            if not names:
                self.bot.send_message(chat_id, "📚 저장된 참고자료가 없습니다.\n파일 보낼 때 캡션에 '저장해줘' 또는 '저장해줘 문서이름'이라고 적어서 등록하세요.")
                return
            listing = "\n".join(f"- {n}" for n in names)
            self.bot.send_message(chat_id, f"📚 저장된 참고자료 목록:\n{listing}")

        @self.bot.message_handler(commands=['forget'])
        def handle_forget(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            if not self.knowledge_store:
                self.bot.send_message(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.")
                return
            name = message.text.partition(' ')[2].strip()
            if not name:
                self.bot.send_message(chat_id, "사용법: /forget 문서이름")
                return
            if self.knowledge_store.remove(chat_id, name):
                self.user_sessions.pop(chat_id, None)
                self.bot.send_message(chat_id, f"🗑 '{name}' 자료를 삭제했습니다.")
            else:
                self.bot.send_message(chat_id, f"'{name}'이라는 이름의 저장된 자료를 찾지 못했습니다. /kb로 목록을 확인하세요.")

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            quick_list = "\n".join(f"/{c} - {t[:28]}..." for c, t in self.quick_commands.items())
            quick_section = f"\n\n[전문 분야 단축 명령어]\n{quick_list}" if quick_list else ""
            kb_section = (
                "\n\n[영구 참고자료]\n"
                "파일 보낼 때 캡션에 '저장해줘' 또는 '저장해줘 문서이름'이라고 적으면 영구 저장됩니다 (reset해도 안 사라짐).\n"
                "/kb - 저장된 자료 목록 확인\n"
                "/forget 문서이름 - 저장된 자료 삭제"
                if self.knowledge_store else ""
            )
            self.bot.send_message(
                message.chat.id,
                f"{self.welcome_message}\n\n"
                "/search on|off - 최신 정보 검색 기능 켜기/끄기 (기본 꺼짐)\n"
                "/reset - 대화 기록 초기화\n"
                "/model - 현재 사용 모델 확인\n"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/myid - 내 chat_id 확인\n"
                "/help - 이 도움말 보기"
                f"{quick_section}{kb_section}"
            )

        @self.bot.message_handler(commands=['start', 'reset'])
        def handle_commands(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            command = message.text.split()[0].lower()
            if command == '/start':
                self.bot.send_message(chat_id, self.welcome_message, parse_mode='Markdown')
            elif command == '/reset':
                self.user_sessions.pop(chat_id, None)
                self.store.clear(chat_id)
                self.bot.send_message(chat_id, "🔄 대화 기록이 초기화되었습니다. (영구 참고자료는 유지됩니다)")

        for cmd_name, template in self.quick_commands.items():
            self.bot.message_handler(commands=[cmd_name])(self._make_quick_command_handler(cmd_name, template))

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self._process_and_reply(chat_id, message.text, message.text)

        @self.bot.message_handler(content_types=['document', 'photo'])
        def handle_file(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return

            caption = message.caption or ""

            if message.content_type == 'document':
                fallback_name = message.document.file_name or "문서"
            else:
                fallback_name = "사진"

            remember_name = extract_remember_name(caption, fallback_name)

            self.bot.send_chat_action(chat_id, 'typing')
            placeholder = self.bot.send_message(chat_id, SAVING_MESSAGE if remember_name else THINKING_MESSAGE)
            try:
                if message.content_type == 'document':
                    file_name = message.document.file_name or ""
                    mime_type = message.document.mime_type or "application/octet-stream"
                    file_info = self.bot.get_file(message.document.file_id)
                    file_bytes = self.bot.download_file(file_info.file_path)

                    if file_name.lower().endswith((".xlsx", ".xls")):
                        excel_text = self._excel_to_text(file_bytes)
                        if remember_name:
                            if self.knowledge_store:
                                self.knowledge_store.add(chat_id, remember_name, excel_text)
                                self.user_sessions.pop(chat_id, None)
                                self._reply(chat_id, f"📚 '{remember_name}' 자료로 저장했습니다.", edit_message_id=placeholder.message_id)
                            else:
                                self._show_error(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.", edit_message_id=placeholder.message_id)
                            return
                        content_parts = [excel_text, "이 파일의 내용을 분석하고 핵심을 요약해줘."]
                    else:
                        prompt = EXTRACTION_PROMPT if remember_name else "이 파일의 내용을 분석하고 핵심을 요약해줘."
                        content_parts = [types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt]
                else:
                    file_id = message.photo[-1].file_id
                    file_info = self.bot.get_file(file_id)
                    file_bytes = self.bot.download_file(file_info.file_path)
                    prompt = EXTRACTION_PROMPT if remember_name else "이 파일의 내용을 분석하고 핵심을 요약해줘."
                    content_parts = [types.Part.from_bytes(data=file_bytes, mime_type="image/jpeg"), prompt]

                response = self._send_with_retry(chat_id, content_parts)
                reply_text = response.text or EMPTY_REPLY_FALLBACK

                if remember_name:
                    if self.knowledge_store:
                        self.knowledge_store.add(chat_id, remember_name, reply_text)
                        self.user_sessions.pop(chat_id, None)
                        self._reply(chat_id, f"📚 '{remember_name}' 자료로 저장했습니다. 앞으로 관련 질문에 항상 참고합니다.", edit_message_id=placeholder.message_id)
                    else:
                        self._show_error(chat_id, "이 봇은 영구 자료 저장 기능이 없습니다.", edit_message_id=placeholder.message_id)
                else:
                    self._reply(chat_id, reply_text, edit_message_id=placeholder.message_id)
                    self._save_history_safely(chat_id, f"[파일 첨부] {caption or '분석 요청'}", reply_text)
            except Exception as e:
                logger.error(f"[{self.name}] 파일 처리 예외 (Chat ID: {chat_id}): {e}", exc_info=True)
                self._show_error(
                    chat_id,
                    "⚠️ 파일 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    edit_message_id=placeholder.message_id
                )

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
