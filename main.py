import os
import time
import sqlite3
import logging
import threading
from typing import Dict, Any, List, Optional, Union
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
# 헬스체크 서버
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

TELEGRAM_MAX_LEN = 4000

def split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> List[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind('\n', 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks

# ==================================================================
# Gemini 라우터
# ==================================================================
class GeminiRouter:
    FALLBACK_MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    EXCLUDE_KEYWORDS = [
        "tts", "audio", "image", "vision", "embedding",
        "aqa", "live", "veo", "imagen", "learnlm",
    ]

    def __init__(self, client: genai.Client):
        self.client = client
        self._lock = threading.Lock()
        self.models = self._discover_models() or list(self.FALLBACK_MODELS)
        self.model_index = 0
        self.model_name = self.models[0]
        logger.info(f"사용 가능한 모델 {len(self.models)}개 확인, 1순위로 시작: {self.model_name}")

    def _discover_models(self) -> List[str]:
        try:
            raw_models = list(self.client.models.list())
        except Exception as e:
            logger.warning(f"모델 목록 조회 실패, 기본 후보 목록 사용: {e}")
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
        ordered = latest_flash + other_flash + others
        logger.info(f"실시간 조회된 사용 가능 모델(우선순위 정렬): {ordered[:6]}{'...' if len(ordered) > 6 else ''}")
        return ordered

    @staticmethod
    def is_retryable_model_error(e: Exception) -> bool:
        msg = str(e)
        return any(code in msg for code in ("RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404"))

    def advance_model(self) -> bool:
        with self._lock:
            if self.model_index + 1 < len(self.models):
                self.model_index += 1
                self.model_name = self.models[self.model_index]
                logger.warning(f"모델 사용 불가(할당량 초과 또는 미지원)로 전환 → {self.model_name}")
                return True
            return False

    def generate(self, contents, config=None, max_transient_retries: int = 2):
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
                    if self.advance_model():
                        transient_left = max_transient_retries
                        continue
                    raise last_err
                transient_left -= 1
                if transient_left < 0:
                    raise last_err
                wait = 1.5 * (max_transient_retries - transient_left)
                logger.warning(f"Gemini 호출 실패, {wait:.1f}초 후 재시도: {e}")
                time.sleep(wait)

    def create_chat(self, history=None, config=None):
        return self.client.chats.create(model=self.model_name, config=config, history=history or [])


router = GeminiRouter(_client)

# ==================================================================
# 대화 기록 저장소
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

    def load_history(self, chat_id: int, limit: int = 20) -> List[Dict[str, str]]:
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
# 1) 번역봇 (영어 / 중국 / 인도네시아 / 베트남 / 태국)
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
    {
        "name": "너구리_베트남",
        "token_env": "TELEGRAM_TOKEN_VI",
        "instruction": (
            "너는 한국어-베트남어 양방향 번역 전용 엔진이다. 잡담, 인사말, 이모지, "
            "부가 설명, 코멘트를 절대 추가하지 말고 오직 정확한 번역문만 출력하라.\n"
            "- 입력이 한국어면: 가장 정확하고 격식 있는 베트남어(공손한 존칭 표현 사용)로 번역하라.\n"
            "- 입력이 베트남어면: 가장 정확한 한국어 존댓말(합니다/습니다체)로 번역하라.\n"
            "출력은 번역문 한 가지만, 다른 텍스트는 절대 포함하지 마라."
        ),
    },
    {
        "name": "너구리_태국",
        "token_env": "TELEGRAM_TOKEN_TH",
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

        @self.bot.message_handler(commands=['reset'])
        def handle_reset(message: Message):
            self.bot.send_message(
                message.chat.id,
                "ℹ️ 이 봇은 매번 새로운 문장을 독립적으로 번역하기 때문에, "
                "따로 초기화할 대화 기록이 없습니다. 그냥 이어서 번역할 문장을 보내주세요."
            )

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            self.bot.send_message(
                message.chat.id,
                "사용법: 문장을 그대로 보내면 번역문만 반환합니다.\n"
                "이 봇은 대화 기억이 없어 매번 새로 번역합니다.\n"
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
# 2) 스마트 개인비서 봇 (똑똑한 너구리)
# ==================================================================
SMART_BOT_INSTRUCTION = (
    "너는 '너구리'라는 이름의 유능한 개인 비서다. 사용자의 업무, 재무, 학습, 일상 질문을 폭넓게 돕는다.\n"
    "- 복잡하거나 계산이 필요한 질문은 단계적으로 사고 과정을 거쳐 정확하게 검산한 뒤 답하라.\n"
    "- 질문이 모호하면 답을 짐작하지 말고 먼저 되물어서 명확히 하라.\n"
    "- 파일(엑셀, PDF, 이미지 등)이 첨부되면 내용을 꼼꼼히 분석해 핵심을 정리하고, "
    "이상하거나 비정상적인 값이 있으면 짚어줘라.\n"
    "- 답변은 불필요하게 장황하지 않게, 필요하면 표나 목록을 사용해 간결하고 실용적으로 작성하라."
)

SMART_BOT_CONFIG_BASE = types.GenerateContentConfig(system_instruction=SMART_BOT_INSTRUCTION)
SMART_BOT_CONFIG_SEARCH = types.GenerateContentConfig(
    system_instruction=SMART_BOT_INSTRUCTION,
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

class SmartGeminiBot:
    def __init__(self, token: str, store: ChatHistoryStore):
        self.bot = telebot.TeleBot(token)
        self.store = store
        self.user_sessions: Dict[int, Any] = {}
        self.search_enabled: Dict[int, bool] = {}
        self._register_handlers()

    def _history_to_genai_format(self, rows: List[Dict[str, str]]) -> List[types.Content]:
        return [types.Content(role=r["role"], parts=[types.Part(text=r["content"])]) for r in rows]

    def _get_chat_session(self, chat_id: int):
        if chat_id not in self.user_sessions:
            history = self._history_to_genai_format(self.store.load_history(chat_id))
            search_on = self.search_enabled.get(chat_id, False)
            config = SMART_BOT_CONFIG_SEARCH if search_on else SMART_BOT_CONFIG_BASE
            self.user_sessions[chat_id] = router.create_chat(history=history, config=config)
            logger.info(
                f"[똑똑한 너구리] 세션 생성/복원: Chat ID {chat_id} "
                f"(기록 {len(history)}건, 검색={'ON' if search_on else 'OFF'}, 모델={router.model_name})"
            )
        return self.user_sessions[chat_id]

    def _send_with_retry(self, chat_id: int, content: Union[str, list], max_transient_retries: int = 2):
        transient_left = max_transient_retries
        last_err = None
        while True:
            chat_session = self._get_chat_session(chat_id)
            try:
                return chat_session.send_message(content)
            except Exception as e:
                last_err = e
                if router.is_retryable_model_error(e):
                    if router.advance_model():
                        self.user_sessions.pop(chat_id, None)
                        transient_left = max_transient_retries
                        continue
                    raise last_err
                transient_left -= 1
                if transient_left < 0:
                    raise last_err
                wait = 1.5 * (max_transient_retries - transient_left)
                logger.warning(f"[똑똑한 너구리] 응답 실패, {wait:.1f}초 후 재시도: {e}")
                time.sleep(wait)

    def _reply(self, chat_id: int, text: str):
        for chunk in split_message(text):
            try:
                self.bot.send_message(chat_id, chunk, parse_mode='Markdown')
            except Exception:
                self.bot.send_message(chat_id, chunk)

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            self.bot.send_message(message.chat.id, f"🆔 chat_id: `{message.chat.id}`", parse_mode='Markdown')

        @self.bot.message_handler(commands=['model'])
        def handle_model(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"🧠 현재 사용 중인 모델: `{router.model_name}`", parse_mode='Markdown')

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
            self.bot.send_message(
                message.chat.id,
                "사용법: 궁금한 걸 물어보면 이전 대화를 기억하며 답합니다.\n"
                "엑셀/PDF/사진을 보내면 분석해줍니다.\n"
                "/search on|off - 최신 정보 검색 기능 켜기/끄기 (기본 꺼짐)\n"
                "/reset - 대화 기록 초기화\n"
                "/model - 현재 사용 모델 확인\n"
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
                    "🚀 *똑똑한 너구리 가동*\n\n"
                    "이전 대화 문맥을 기억합니다. 엑셀·PDF·사진을 보내면 분석도 해드립니다.\n"
                    "최신 정보 검색이 필요하면 `/search on`으로 켜세요 (평소엔 꺼두는 걸 추천).\n"
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
                response = self._send_with_retry(chat_id, user_input)
                reply_text = response.text

                self.store.append(chat_id, "user", user_input)
                self.store.append(chat_id, "model", reply_text)
                self._reply(chat_id, reply_text)
            except Exception as e:
                logger.error(f"[똑똑한 너구리] 예외 발생 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "⚠️ 오류가 발생했습니다. 잠시 후 다시 시도하거나 `/reset`을 입력해 주세요.")

        @self.bot.message_handler(content_types=['document', 'photo'])
        def handle_file(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "⛔ 승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                if message.content_type == 'document':
                    file_id = message.document.file_id
                    mime_type = message.document.mime_type or "application/octet-stream"
                else:
                    file_id = message.photo[-1].file_id
                    mime_type = "image/jpeg"

                file_info = self.bot.get_file(file_id)
                file_bytes = self.bot.download_file(file_info.file_path)
                caption = message.caption or "이 파일의 내용을 분석하고 핵심을 요약해줘."

                response = self._send_with_retry(
                    chat_id,
                    [types.Part.from_bytes(data=file_bytes, mime_type=mime_type), caption]
                )
                reply_text = response.text

                self.store.append(chat_id, "user", f"[파일 첨부] {caption}")
                self.store.append(chat_id, "model", reply_text)
                self._reply(chat_id, reply_text)
            except Exception as e:
                logger.error(f"[똑똑한 너구리] 파일 처리 예외 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "⚠️ 파일 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)


# ==================================================================
# 실행: 폴링이 죽어도 자동으로 재시작
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
