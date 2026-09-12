import logging

import telebot
from telebot.types import Message
from google.genai import types

from common import TelegramBotBase, log_token_usage, split_message
from router import GeminiRouter

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # Telegram Bot API가 이보다 큰 파일은 getFile로 받을 수 없음


class FileTooLargeError(Exception):
    pass


DOCUMENT_INSTRUCTION = (
    "너는 사업주가 올린 서류 사진/문서를 읽고 그 안의 정보를 정리해서 알려주는 AI다. 이 대화에만 "
    "보여주는 용도이며, 어디에도 저장하거나 전달하지 않는다.\n\n"
    "받은 문서가 통장/계좌 관련 서류면 [계좌 정보]로, 가족관계증명서면 [가족관계증명서]로 제목을 "
    "붙이고 아래 형식으로 정리하라:\n\n"
    "[계좌 정보]\n"
    "은행명: ...\n"
    "계좌번호: ...\n"
    "예금주: ...\n\n"
    "[가족관계증명서]\n"
    "본인: 이름 / 생년월일 / 주민등록번호\n"
    "가족: (관계) 이름 / 생년월일 / 주민등록번호  ← 증명서에 나온 가족 구성원마다 한 줄씩 반복\n\n"
    "주민등록번호는 문서에 보이는 그대로 전체 숫자를 다 적어라(마스킹하지 마라). 글씨가 흐리거나 "
    "잘려서 안 보이는 항목은 추측하지 말고 '판독 불가'라고 써라. 위 두 종류 서류가 아니거나 서류 "
    "종류를 알 수 없으면 그렇게 말하고 보이는 정보만 정리하라.\n\n"
    "마크다운 특수기호와 이모지는 쓰지 말아라."
)

DOCUMENT_WELCOME = (
    "서류확인 너구리 가동\n\n"
    "계좌 관련 서류나 가족관계증명서 사진(또는 PDF)을 보내면 내용을 읽어서 정리해드립니다.\n"
    "따로 저장하지 않고 이 대화에만 보여드려요."
)


class NeoguriDocumentBot(TelegramBotBase):
    def __init__(self, name: str, token: str, router: GeminiRouter):
        self.name = name
        self.bot = telebot.TeleBot(token, threaded=False)
        self.router = router
        self.config = types.GenerateContentConfig(
            system_instruction=DOCUMENT_INSTRUCTION,
            temperature=0.1,  # 서류 판독이라 창의성보다 정확한 재현이 중요
            max_output_tokens=1200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # 단순 판독이라 추론 불필요
        )
        self._register_handlers()

    def _download_file_payload(self, message: Message):
        if message.content_type == 'document':
            mime_type = message.document.mime_type or "application/octet-stream"
            file_size = message.document.file_size
            file_id = message.document.file_id
        else:
            mime_type = "image/jpeg"
            file_size = message.photo[-1].file_size
            file_id = message.photo[-1].file_id

        if file_size and file_size > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"파일이 너무 큽니다 ({file_size / 1024 / 1024:.1f}MB). "
                f"최대 {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB까지 가능합니다."
            )

        file_info = self.bot.get_file(file_id)
        file_bytes = self.bot.download_file(file_info.file_path)
        return file_bytes, mime_type

    def _analyze_and_reply(self, message: Message, chat_id: int):
        self.bot.send_chat_action(chat_id, 'typing')
        file_bytes, mime_type = self._download_file_payload(message)
        response = self.router.generate(
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                "이 서류에서 정보를 읽어서 정리해줘.",
            ],
            config=self.config,
        )
        log_token_usage(self.name, response)
        text = response.text or "내용을 읽지 못했습니다."
        for chunk in split_message(text):
            self.bot.send_message(chat_id, chunk)

    def _register_handlers(self):
        self._register_common_handlers()

        @self.bot.message_handler(commands=['start', 'help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(chat_id, DOCUMENT_WELCOME)

        @self.bot.message_handler(content_types=['photo', 'document'])
        def handle_file(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self._ai_cooldown_ok(chat_id):
                return
            try:
                self._analyze_and_reply(message, chat_id)
            except FileTooLargeError as e:
                self.bot.send_message(chat_id, str(e))
            except Exception as e:
                logger.error(f"[{self.name}] 서류 처리 실패 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "서류를 처리하는 중 오류가 발생했습니다.")

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(chat_id, "계좌 서류나 가족관계증명서 사진(또는 PDF)을 보내주세요.")

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
