import logging
import telebot
from telebot.types import Message
from google.genai import types

from common import log_token_usage, TelegramBotBase
from router import GeminiRouter

logger = logging.getLogger(__name__)

EMPTY_REPLY_FALLBACK = "응답이 비어 있습니다. 다시 한번 보내주세요."

# 모든 번역봇이 공유하는 규칙 — 언어별 지시문에서 반복하지 않도록 한 곳에서만 정의한다.
TRANSLATOR_COMMON_RULE = "번역문 외에 잡담·인사말·이모지·부가 설명·코멘트를 절대 추가하지 마라."

TRANSLATOR_BOT_DEFS = [
    {
        "name": "너구리_영어", "token_env": "TELEGRAM_TOKEN_EN",
        "instruction": (
            f"너는 한국어-영어 양방향 번역 엔진이다. {TRANSLATOR_COMMON_RULE}\n"
            "한국어→격식 있는 영어로, 영어→한국어 존댓말(합니다체)로 번역하라."
        ),
    },
    {
        "name": "너구리_중국", "token_env": "TELEGRAM_TOKEN_ZH",
        "instruction": (
            f"너는 한국어-중국어(간체) 양방향 번역 엔진이다. {TRANSLATOR_COMMON_RULE}\n"
            "한국어→격식 있는 중국어(您 등 존칭)로 번역 후 번역문 뒤 괄호에 병음을 표기하라. "
            "중국어→한국어 존댓말(합니다체)로 번역하라."
        ),
    },
    {
        "name": "너구리_인도네시아", "token_env": "TELEGRAM_TOKEN_ID",
        "instruction": (
            f"너는 한국어-인도네시아어 양방향 번역 엔진이다. {TRANSLATOR_COMMON_RULE}\n"
            "한국어→격식 있는 인도네시아 표준어(Bahasa Baku)로, 인도네시아어→한국어 존댓말(합니다체)로 번역하라."
        ),
    },
    {
        "name": "너구리_베트남", "token_env": "TELEGRAM_TOKEN_VI",
        "instruction": (
            f"너는 한국어-베트남어 양방향 번역 엔진이다. {TRANSLATOR_COMMON_RULE}\n"
            "한국어→격식 있는 베트남어로, 베트남어→한국어 존댓말(합니다체)로 번역하라."
        ),
    },
    {
        "name": "너구리_태국", "token_env": "TELEGRAM_TOKEN_TH",
        "instruction": (
            f"너는 한국어-태국어 양방향 번역 엔진이다. {TRANSLATOR_COMMON_RULE}\n"
            "한국어→격식 있는 태국어(공손한 어미 ครับ/ค่ะ)로, 태국어→한국어 존댓말(합니다체)로 번역하라."
        ),
    },
]

class NeoguriTranslatorBot(TelegramBotBase):
    def __init__(self, name: str, token: str, instruction: str, router: GeminiRouter):
        self.name = name
        self.bot = telebot.TeleBot(token, threaded=False)
        self.router = router
        self.config = types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.2,  # 번역은 창의적 변주보다 일관되고 정확한 결과가 중요
            max_output_tokens=2048,
        )
        self._register_handlers()

    def _register_handlers(self):
        self._register_common_handlers()

        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(chat_id, f"{self.name} 준비되었습니다. 번역할 문장을 보내주세요.")

        @self.bot.message_handler(commands=['reset'])
        def handle_reset(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(
                chat_id,
                "이 봇은 매번 새로운 문장을 독립적으로 번역하기 때문에, "
                "따로 초기화할 대화 기록이 없습니다. 그냥 이어서 번역할 문장을 보내주세요."
            )

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(
                chat_id,
                "사용법: 문장을 그대로 보내면 번역문만 반환합니다.\n"
                "/myid - 내 chat_id 확인\n"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/help - 이 도움말 보기"
            )

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                response = self.router.generate(contents=message.text, config=self.config)
                log_token_usage(self.name, response)
                reply_text = response.text or EMPTY_REPLY_FALLBACK
                self.bot.send_message(chat_id, reply_text)
            except Exception as e:
                logger.error(f"[{self.name}] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
