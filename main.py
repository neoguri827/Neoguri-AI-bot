import os
import time
import sqlite3
import logging
import threading
from typing import Dict, Any, List, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
from contextlib import closing

import telebot
from telebot.types import Message
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================================
# 헬스체크 서버 (Render 슬립 방지/상태 확인용, 전체 봇 공용 1개)
# ==================================================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "All Neoguri Bots Running"}')

    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()

threading.Thread(target=start_health_server, daemon=True).start()

# ==================================================================
# 공용 설정 / 화이트리스트
# ==================================================================
def _parse_allowed_ids(raw: Optional[str]) -> set:
    if not raw:
        return set()
    return {int(t.strip()) for t in raw.split(",") if t.strip().lstrip("-").isdigit()}

ALLOWED_CHAT_IDS = _parse_allowed_ids(os.environ.get("ALLOWED_CHAT_IDS"))

def is_allowed(chat_id: int) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    return chat_id in ALLOWED_CHAT_IDS

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 없습니다.")

_client = genai.Client(api_key=GEMINI_API_KEY)

# ==================================================================
# Gemini 라우터: 사용 가능한 모델을 자동으로 찾고, 실패 시 재시도까지 처리
# ==================================================================
class GeminiRouter:
    PREFERRED_MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    def __init__(self, client: genai.Client):
        self.client = client
        self.model_name = self._resolve_model()

    def _resolve_model(self) -> str:
        for name in self.PREFERRED_MODELS:
            try:
                self.client.models.generate_content(model=name, contents="ping")
                logger.info(f"사용할 Gemini 모델 확정: {name}")
                return name
            except Exception as e:
                logger.warning(f"모델 '{name}' 사용 불가, 다음 후보로 전환: {e}")
        raise RuntimeError(
            "사용 가능한 Gemini 모델이 하나도 없습니다. GEMINI_API_KEY가 유효한지 확인하세요."
        )

    def generate(self, contents, config=None, retries: int = 2):
        last_err = None
        for attempt in range(retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            except Exception as e:
                last_err = e
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    logger.warning(f"Gemini 호출 실패, {wait:.1f}초 후 재시도({attempt+1}/{retries}): {e}")
                    time.sleep(wait)
        raise last_err

    def create_chat(self, history=None):
        return self.client.chats.create(model=self.model_name, history=history or [])


router = GeminiRouter(_client)

# ==================================================================
# 대화 기록 저장소 (스마트 비서 봇 전용, 사용자별 최근 200개만 보관)
# ==================================================================
DB_PATH = os.environ.get("DB_PATH", "chat_history.db")

class ChatHistoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with closing(self._get_conn()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON messages(chat_id)")
            conn.commit()

    def load_history(self, chat_id: int, limit: int = 40) -> List[Dict[str, str]]:
        with closing(self._get_conn()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit)
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def append(self, chat_id: int, role: str, content: str, max_rows: int = 200):
        with closing(self._get_conn()) as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content)
            )
            conn.execute("""
                DELETE FROM messages
                WHERE chat_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?
                )
            """, (chat_id, chat_id, max_rows))
            conn.commit()

    def clear(self, chat_id: int):
        with closing(self._get_conn()) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.commit()

# ==================================================================
# 1) 번역봇 (너구리_영어 / 중국 / 인도네시아)
#    - 정확한 번역만 출력, 잡담/이모지/코멘트 일체 금지, 최대 존댓말·격식체
# ==================================================================
TRANSLATOR_BOT_DEFS = [
    {
        "name": "너구리_영어",
        "token_env": "TELEGRAM_TOKEN_EN",
        "instruction": (
            "너는 한국어-영어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, 부가 설명, "
            "코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 정확하고 격식 있는(formal) 영어로 번역하라. "
            "구어체 축약형(don't, can't 등) 대신 정중한 표현을 사용하라.\n"
            "- 입력이 영어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
    {
        "name": "너구리_중국",
        "token_env": "TELEGRAM_TOKEN_ZH",
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
        "name": "너구리_인도네시아",
        "token_env": "TELEGRAM_TOKEN_ID",
        "instruction": (
            "너는 한국어-인도네시아어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, "
            "부가 설명, 코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 격식 있는 인도네시아 표준어(Bahasa Baku, 공손한 표현)로 번역하라. "
            "casual/Gaul체는 절대 사용하지 마라.\n"
            "- 입력이 인도네시아어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
]

class NeoguriTranslatorBot:
    def __init__(self, name: str, token: str, instruction: str):
        self.name = name
        self.bot = telebot.TeleBot(token)
        self.config = types.GenerateContentConfig(system_instruction=instruction)
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            self.bot.send_message(message.chat.id, f"🆔 chat_id: `{message.chat.id}`", parse_mode='Markdown')

        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            self.bot.send_message(message.chat.id, f"{self.name} 준비되었습니다. 번역할 문장을 보내주세요.")

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            self.bot.send_message(
                message.chat.id,
                "사용법: 문장을 그대로 보내면 번역문만 반환합니다.\n"
                "/myid - 내 chat_id 확인\n"
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
                response = router.generate(contents=message.text, config=self.config)
                self.bot.send_message(chat_id, response.text)
            except Exception as e:
                logger.error(f"[{self.name}] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "⚠️ 번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)


# ==================================================================
# 2) 스마트 개인비서 봇 (똑똑한 너구리) - 대화 기억 있음
# ==================================================================
class SmartGeminiBot:
    def __init__(self, token: str, store: ChatHistoryStore):
        self.bot = telebot.TeleBot(token)
        self.store = store
        self.user_sessions: Dict[int, Any] = {}
        self._register_handlers()

    def _history_to_genai_format(self, rows: List[Dict[str, str]]) -> List[types.Content]:
        return [types.Content(role=r["role"], parts=[types.Part(text=r["content"])]) for r in rows]

    def _get_chat_session(self, chat_id: int):
        if chat_id not in self.user_sessions:
            history = self._history_to_genai_format(self.store.load_history(chat_id))
            self.user_sessions[chat_id] = router.create_chat(history=history)
            logger.info(f"[똑똑한 너구리] 세션 생성/복원: Chat ID {chat_id} (기록 {len(history)}건)")
        return self.user_sessions[chat_id]

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            self.bot.send_message(message.chat.id, f"🆔 chat_id: `{message.chat.id}`", parse_mode='Markdown')

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            self.bot.send_message(
                message.chat.id,
                "사용법: 궁금한 걸 물어보면 이전 대화를 기억하며 답합니다.\n"
                "/reset - 대화 기록 초기화\n"
                "/myid - 내 chat_id 확인\n"
                "/help - 이 도움말 보기"
            )

        @self.bot.message_handler(commands=['start', 'reset'])
        def handle_commands(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            command = message.text.split()[0].lower()
            if command == '/start':
                self.bot.send_message(
                    chat_id,
                    "🚀 *똑똑한 너구리 가동*\n\n이전 대화 문맥을 기억합니다.\n"
                    "새 주제로 시작하려면 `/reset`을 입력하세요.",
                    parse_mode='Markdown'
                )
            elif command == '/reset':
                self.user_sessions.pop(chat_id, None)
                self.store.clear(chat_id)
                self.bot.send_message(chat_id, "🔄 대화 기록이 초기화되었습니다.")

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            user_input = message.text
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                chat_session = self._get_chat_session(chat_id)
                response = self._send_with_retry(chat_session, user_input)
                reply_text = response.text

                self.store.append(chat_id, "user", user_input)
                self.store.append(chat_id, "model", reply_text)

                try:
                    self.bot.send_message(chat_id, reply_text, parse_mode='Markdown')
                except Exception:
                    self.bot.send_message(chat_id, reply_text)
            except Exception as e:
                logger.error(f"[똑똑한 너구리] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "⚠️ 오류가 발생했습니다. 잠시 후 다시 시도하거나 `/reset`을 입력해 주세요.")

    def _send_with_retry(self, chat_session, text: str, retries: int = 2):
        last_err = None
        for attempt in range(retries + 1):
            try:
                return chat_session.send_message(text)
            except Exception as e:
                last_err = e
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    logger.warning(f"[똑똑한 너구리] 응답 실패, {wait:.1f}초 후 재시도: {e}")
                    time.sleep(wait)
        raise last_err

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)


# ==================================================================
# 실행: 폴링이 죽어도 자동으로 재시작하는 래퍼
# ==================================================================
def run_forever(bot_obj, name: str):
    while True:
        try:
            logger.info(f"[{name}] 폴링 시작")
            bot_obj.run()
        except Exception as e:
            logger.error(f"[{name}] 폴링이 예외로 중단됨, 5초 후 재시작: {e}", exc_info=True)
            time.sleep(5)


def main():
    threads = []

    for cfg in TRANSLATOR_BOT_DEFS:
        token = os.environ.get(cfg["token_env"])
        if not token:
            logger.warning(f"{cfg['token_env']} 없어서 {cfg['name']} 건너뜁니다.")
            continue
        bot_obj = NeoguriTranslatorBot(cfg["name"], token, cfg["instruction"])
        t = threading.Thread(target=run_forever, args=(bot_obj, cfg["name"]), daemon=True)
        t.start()
        threads.append(t)

    smart_token = os.environ.get("TELEGRAM_TOKEN_SMART")
    if smart_token:
        store = ChatHistoryStore(DB_PATH)
        bot_obj = SmartGeminiBot(smart_token, store)
        t = threading.Thread(target=run_forever, args=(bot_obj, "똑똑한 너구리"), daemon=True)
        t.start()
        threads.append(t)
    else:
        logger.warning("TELEGRAM_TOKEN_SMART 없어서 똑똑한 너구리는 건너뜁니다.")

    if not threads:
        raise ValueError("실행 가능한 봇이 없습니다. 환경변수를 확인하세요.")

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
