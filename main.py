import os
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import telebot
from telebot.types import Message
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 헬스체크 서버 (Render 슬립 방지용, 3개 봇 공용으로 1개만 실행)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "Neoguri Translator Bots Running"}')

    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()

threading.Thread(target=start_health_server, daemon=True).start()

# 화이트리스트 (기존 개인비서 봇과 동일한 방식, 3개 봇 공용)
def _parse_allowed_ids(raw: Optional[str]) -> set:
    if not raw:
        return set()
    return {int(t.strip()) for t in raw.split(",") if t.strip().lstrip("-").isdigit()}

ALLOWED_CHAT_IDS = _parse_allowed_ids(os.environ.get("ALLOWED_CHAT_IDS"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 없습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3-flash-preview"

# 너구리 번역봇 정의 (이름 / 토큰 환경변수 이름 / 번역 지침)
BOT_DEFS = [
    {
        "name": "너구리_영어",
        "token_env": "TELEGRAM_TOKEN_EN",
        "instruction": (
            "너는 한국어-영어 양방향 번역 전문가 '너구리'야. "
            "사용자가 한국어를 입력하면 자연스러운 영어로 번역하고, "
            "영어를 입력하면 자연스러운 한국어로 번역해줘. "
            "번역문을 가장 먼저 보여주고, 어색하거나 오해될 수 있는 표현이 있으면 "
            "짧게 한 줄 코멘트를 덧붙여도 좋아. 말투는 친근하게, "
            "필요하면 🦝 이모지를 가끔 사용해."
        ),
    },
    {
        "name": "너구리_중국",
        "token_env": "TELEGRAM_TOKEN_ZH",
        "instruction": (
            "너는 한국어-중국어(간체) 양방향 번역 전문가 '너구리'야. "
            "사용자가 한국어를 입력하면 자연스러운 중국어로 번역하고 "
            "괄호 안에 병음(pinyin)을 함께 적어줘. "
            "중국어를 입력하면 자연스러운 한국어로 번역해줘. "
            "말투는 친근하게, 필요하면 🦝 이모지를 가끔 사용해."
        ),
    },
    {
        "name": "너구리_인도네시아",
        "token_env": "TELEGRAM_TOKEN_ID",
        "instruction": (
            "너는 한국어-인도네시아어 양방향 번역 전문가 '너구리'야. "
            "사용자가 한국어를 입력하면 인도네시아어로 번역하되, "
            "격식체(Baku)와 일상체(Gaul) 두 버전을 모두 보여줘. "
            "인도네시아어를 입력하면 자연스러운 한국어로 번역해줘. "
            "말투는 친근하게, 필요하면 🦝 이모지를 가끔 사용해."
        ),
    },
]

class NeoguriTranslatorBot:
    def __init__(self, name: str, token: str, instruction: str):
        self.name = name
        self.bot = telebot.TeleBot(token)
        self.config = types.GenerateContentConfig(system_instruction=instruction)
        self._register_handlers()

    def _is_allowed(self, chat_id: int) -> bool:
        if not ALLOWED_CHAT_IDS:
            return True
        return chat_id in ALLOWED_CHAT_IDS

    def _register_handlers(self):

        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            self.bot.send_message(
                message.chat.id, f"🆔 chat_id: `{message.chat.id}`", parse_mode='Markdown'
            )

        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            self.bot.send_message(
                message.chat.id, f"🦝 안녕! 나는 {self.name}야. 텍스트를 보내면 바로 번역해줄게!"
            )

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not self._is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return

            self.bot.send_chat_action(chat_id, 'typing')
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=message.text,
                    config=self.config,
                )
                self.bot.send_message(chat_id, response.text)
            except Exception as e:
                logger.error(f"[{self.name}] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "⚠️ 번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    def run(self):
        logger.info(f"[{self.name}] 폴링 시작")
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)


def main():
    threads = []
    for cfg in BOT_DEFS:
        token = os.environ.get(cfg["token_env"])
        if not token:
            logger.warning(f"{cfg['token_env']} 환경변수가 없어서 {cfg['name']} 봇은 건너뜁니다.")
            continue
        bot_instance = NeoguriTranslatorBot(cfg["name"], token, cfg["instruction"])
        t = threading.Thread(target=bot_instance.run, daemon=True)
        t.start()
        threads.append(t)

    if not threads:
        raise ValueError("실행 가능한 봇이 없습니다. TELEGRAM_TOKEN_* 환경변수를 확인하세요.")

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
