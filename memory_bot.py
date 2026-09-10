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
from store import ChatHistoryStore

logger = logging.getLogger(__name__)

EMPTY_REPLY_FALLBACK = "⚠️ 응답이 비어 있습니다. 다시 한번 시도해 주세요."
THINKING_MESSAGE = "🤔 답변 준비중입니다..."


class MemoryGeminiBot:
    def __init__(self, name: str, token: str, router: GeminiRouter, store: ChatHistoryStore,
                 base_instruction: str, welcome_message: str,
                 quick_commands: Optional[Dict[str, str]] = None):
        self.name = name
        self.bot = telebot.TeleBot(token)
        self.router = router
        self.store = store
        self.welcome_message = welcome_message
        self.quick_commands = quick_commands or {}
        self.config_base = types.GenerateContentConfig(system_instruction=base_instruction)
        self.config_search = types.GenerateContentConfig(
            system_instruction=base_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        self.user_sessions: Dict[int, Any] = {}
        self.search_enabled: Dict[int, bool] = {}
        self._register_handlers()

    def _history_to_genai_format(self, rows: List[Dict[str, str]]) -> List[types.Content]:
        return [types.Content(role=r["role"], parts=[types.Part(text=r["content"])]) for r in rows]

    def _get_chat_session(self, chat_id: int):
        if chat_id not in self.user_sessions:
            history = self._history_to_genai_format(self.store.load_history(chat_id))
            search_on = self.search_enabled.get(chat_id, False)
            config = self.config_search if search_on else self.config_base
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

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            quick_list = "\n".join(f"/{c} - {t[:28]}..." for c, t in self.quick_commands.items())
            quick_section = f"\n\n[전문 분야 단축 명령어]\n{quick_list}" if quick_list else ""
            self.bot.send_message(
                message.chat.id,
                f"{self.welcome_message}\n\n"
                "/search on|off - 최신 정보 검색 기능 켜기/끄기 (기본 꺼짐)\n"
                "/reset - 대화 기록 초기화\n"
                "/model - 현재 사용 모델 확인\n"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/myid - 내 chat_id 확인\n"
                "/help - 이 도움말 보기"
                f"{quick_section}"
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
                self.bot.send_message(chat_id, "🔄 대화 기록이 초기화되었습니다.")

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
            self.bot.send_chat_action(chat_id, 'typing')
            placeholder = self.bot.send_message(chat_id, THINKING_MESSAGE)
            try:
                caption = message.caption or "이 파일의 내용을 분석하고 핵심을 요약해줘."

                if message.content_type == 'document':
                    file_name = message.document.file_name or ""
                    mime_type = message.document.mime_type or "application/octet-stream"
                    file_info = self.bot.get_file(message.document.file_id)
                    file_bytes = self.bot.download_file(file_info.file_path)

                    if file_name.lower().endswith((".xlsx", ".xls")):
                        excel_text = self._excel_to_text(file_bytes)
                        content_parts = [excel_text, caption]
                    else:
                        content_parts = [types.Part.from_bytes(data=file_bytes, mime_type=mime_type), caption]
                else:
                    file_id = message.photo[-1].file_id
                    file_info = self.bot.get_file(file_id)
                    file_bytes = self.bot.download_file(file_info.file_path)
                    content_parts = [types.Part.from_bytes(data=file_bytes, mime_type="image/jpeg"), caption]

                response = self._send_with_retry(chat_id, content_parts)
                reply_text = response.text or EMPTY_REPLY_FALLBACK
                self._reply(chat_id, reply_text, edit_message_id=placeholder.message_id)
                self._save_history_safely(chat_id, f"[파일 첨부] {caption}", reply_text)
            except Exception as e:
                logger.error(f"[{self.name}] 파일 처리 예외 (Chat ID: {chat_id}): {e}", exc_info=True)
                self._show_error(
                    chat_id,
                    "⚠️ 파일 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    edit_message_id=placeholder.message_id
                )

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
