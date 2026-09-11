import logging
import telebot
from typing import Dict
from telebot.types import Message
from google.genai import types

from common import is_allowed, get_uptime_str, format_token_usage
from router import GeminiRouter

logger = logging.getLogger(__name__)

EMPTY_REPLY_FALLBACK = "⚠️ 응답이 비어 있습니다. 다시 한번 보내주세요."

TRANSLATOR_BOT_DEFS = [
    {
        "name": "너구리_영어", "token_env": "TELEGRAM_TOKEN_EN",
        "instruction": (
            "너는 한국어-영어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, 부가 설명, "
            "코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 정확하고 격식 있는(formal) 영어로 번역하라.\n"
            "- 입력이 영어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
    {
        "name": "너구리_중국", "token_env": "TELEGRAM_TOKEN_ZH",
        "instruction": (
            "너는 한국어-중국어(간체) 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, "
            "부가 설명, 코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 정확하고 격식 있는 중국어(您 등 존칭 사용)로 번역하고, "
            "번역문 바로 뒤 괄호 안에 병음(pinyin)을 표기하라.\n"
            "- 입력이 중국어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문(및 병음)만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
    {
        "name": "너구리_인도네시아", "token_env": "TELEGRAM_TOKEN_ID",
        "instruction": (
            "너는 한국어-인도네시아어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, "
            "부가 설명, 코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 격식 있는 인도네시아 표준어(Bahasa Baku)로 번역하라.\n"
            "- 입력이 인도네시아어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
    {
        "name": "너구리_베트남", "token_env": "TELEGRAM_TOKEN_VI",
        "instruction": (
            "너는 한국어-베트남어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, "
            "부가 설명, 코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 정확하고 격식 있는 베트남어로 번역하라.\n"
            "- 입력이 베트남어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
    {
        "name": "너구리_태국", "token_env": "TELEGRAM_TOKEN_TH",
        "instruction": (
            "너는 한국어-태국어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, "
            "부가 설명, 코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 정확하고 격식 있는 태국어(공손한 어미 ครับ/ค่ะ 사용)로 번역하라.\n"
            "- 입력이 태국어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
]

class NeoguriTranslatorBot:
    def __init__(self, name: str, token: str, instruction: str, router: GeminiRouter):
        self.name = name
        self.bot = telebot.TeleBot(token)
        self.router = router
        self.config = types.GenerateContentConfig(system_instruction=instruction)
        self.show_tokens: Dict[int, bool] = {}
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"🆔 chat_id: {chat_id}")

        @self.bot.message_handler(commands=['uptime'])
        def handle_uptime(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"⏱ 서버 연속 가동 시간: {get_uptime_str()}")

        @self.bot.message_handler(commands=['tokens'])
        def handle_tokens_toggle(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            parts = message.text.split()
            if len(parts) < 2 or parts[1].lower() not in ('on', 'off'):
                current = "켜짐" if self.show_tokens.get(chat_id, False) else "꺼짐"
                self.bot.send_message(chat_id, f"현재 토큰 사용량 표시: {current}\n사용법: /tokens on 또는 /tokens off")
                return
            enable = parts[1].lower() == 'on'
            self.show_tokens[chat_id] = enable
            self.bot.send_message(chat_id, f"🔢 토큰 사용량 표시를 {'켰습니다' if enable else '껐습니다'}.")

        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"{self.name} 준비되었습니다. 번역할 문장을 보내주세요.")

        @self.bot.message_handler(commands=['reset'])
        def handle_reset(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(
                chat_id,
                "ℹ️ 이 봇은 매번 새로운 문장을 독립적으로 번역하기 때문에, "
                "따로 초기화할 대화 기록이 없습니다. 그냥 이어서 번역할 문장을 보내주세요."
            )

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(
                chat_id,
                "사용법: 문장을 그대로 보내면 번역문만 반환합니다.\n"
                "/tokens on|off - 답변마다 토큰 사용량 표시 켜기/끄기 (기본 꺼짐)\n"
                "/myid - 내 chat_id 확인\n"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/help - 이 도움말 보기"
            )

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                response = self.router.generate(contents=message.text, config=self.config)
                reply_text = response.text or EMPTY_REPLY_FALLBACK
                if self.show_tokens.get(chat_id, False):
                    reply_text += format_token_usage(response)
                self.bot.send_message(chat_id, reply_text)
            except Exception as e:
                logger.error(f"[{self.name}] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "⚠️ 번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
