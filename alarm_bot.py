import time
import logging
import threading
import telebot
from telebot.types import Message
from google.genai import types

from common import is_allowed, get_uptime_str, split_message, ALLOWED_CHAT_IDS
from router import GeminiRouter

logger = logging.getLogger(__name__)

ALARM_INTERVAL_SECONDS = 60 * 60  # 1시간마다 자동 브리핑

ALARM_INSTRUCTION = (
    "너는 '알람너구리'라는 시장/뉴스 브리핑 AI다. 반드시 구글 검색 도구를 실제로 호출해서 "
    "방금 확인한 정보만으로 아래 형식을 간결하게 채워라. 검색 없이 기억이나 추측으로 "
    "수치를 채우는 것은 절대 금지다.\n\n"
    "1. 코스피 지수 (현재가, 전일 대비 등락 및 등락률)\n"
    "2. 나스닥 지수 (현재가, 전일 대비 등락 및 등락률)\n"
    "3. 원/달러 환율\n"
    "4. 주요 뉴스 헤드라인 3~5개 (경제·시사 위주, 간단한 한 줄 요약 포함)\n\n"
    "행동 원칙:\n"
    "- 검색 결과에서 확인하지 못한 항목은 절대 숫자를 지어내지 말고 '확인 불가'라고 명시하라.\n"
    "- 불필요한 인사말이나 부연설명 없이 핵심 수치와 헤드라인만 제공하라.\n"
    "- 마크다운 특수기호(*, _, #, ~, ` 등)는 사용하지 말고 순수 텍스트로 답하라. "
    "금액 표시에는 통화 기호(₩, $ 등)만 예외로 허용한다.\n"
    "- 이모지는 절대 사용하지 마라."
)

ALARM_BRIEFING_PROMPT = "지금 기준 코스피, 나스닥, 원/달러 환율, 주요 뉴스 헤드라인을 정리해줘."

ALARM_WELCOME = (
    "알람너구리 가동\n\n"
    "매시간 자동으로 코스피, 나스닥, 원/달러 환율, 주요 뉴스 헤드라인을 정리해서 보내드립니다.\n"
    "지금 바로 확인하고 싶으면 /now를 입력하세요."
)


class NeoguriAlarmBot:
    def __init__(self, name: str, token: str, router: GeminiRouter,
                 interval_seconds: int = ALARM_INTERVAL_SECONDS):
        self.name = name
        self.bot = telebot.TeleBot(token)
        self.router = router
        self.interval_seconds = interval_seconds
        self.config = types.GenerateContentConfig(
            system_instruction=ALARM_INSTRUCTION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        self._scheduler_started = False
        self._scheduler_lock = threading.Lock()
        self._register_handlers()

    def _build_briefing(self) -> str:
        response = self.router.generate(contents=ALARM_BRIEFING_PROMPT, config=self.config)
        self._log_grounding_info(response)
        return response.text or "정보를 가져오지 못했습니다."

    def _log_grounding_info(self, response):
        """실제로 구글 검색을 근거로 답했는지, 어떤 검색어/출처를 썼는지 로그로 남긴다.
        브리핑 내용이 부정확하다는 의심이 들 때 이 로그로 원인(검색 미실행 vs 검색은 했지만
        결과 해석 오류)을 구분할 수 있다."""
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                logger.warning(f"[{self.name}] 응답에 candidate가 없어 검색 근거를 확인할 수 없습니다 (모델: {self.router.model_name})")
                return
            metadata = getattr(candidates[0], "grounding_metadata", None)
            if not metadata:
                logger.warning(
                    f"[{self.name}] 이번 응답은 구글 검색 없이 생성됨(grounding_metadata 없음) — "
                    f"모델이 검색 도구를 호출하지 않고 답했을 가능성이 높습니다 (모델: {self.router.model_name})"
                )
                return
            queries = list(getattr(metadata, "web_search_queries", None) or [])
            sources = []
            for chunk in (getattr(metadata, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                if web is None:
                    continue
                title = getattr(web, "title", None)
                uri = getattr(web, "uri", None)
                sources.append(f"{title} ({uri})" if title else str(uri))
            logger.info(
                f"[{self.name}] 검색 근거 확인 (모델: {self.router.model_name}) — "
                f"검색어: {queries}, 출처 {len(sources)}건: {sources[:5]}"
            )
        except Exception as e:
            logger.warning(f"[{self.name}] 검색 근거 로깅 실패: {e}")

    def _broadcast(self):
        if not ALLOWED_CHAT_IDS:
            logger.warning(f"[{self.name}] 알림 받을 chat_id가 없습니다(ALLOWED_CHAT_IDS 미설정)")
            return
        text = self._build_briefing()
        for chat_id in ALLOWED_CHAT_IDS:
            for chunk in split_message(text):
                try:
                    self.bot.send_message(chat_id, chunk)
                except Exception as e:
                    logger.warning(f"[{self.name}] 브리핑 전송 실패(chat_id={chat_id}): {e}")

    def _scheduler_loop(self):
        while True:
            try:
                self._broadcast()
            except Exception as e:
                logger.error(f"[{self.name}] 정기 브리핑 실패: {e}", exc_info=True)
            time.sleep(self.interval_seconds)

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            # ALLOWED_CHAT_IDS 등록 전에도 본인 chat_id를 확인할 수 있어야 하므로
            # 이 명령만 is_allowed 검사를 우회한다.
            chat_id = message.chat.id
            self.bot.send_message(chat_id, f"chat_id: {chat_id}")

        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, ALARM_WELCOME)

        @self.bot.message_handler(commands=['now'])
        def handle_now(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                text = self._build_briefing()
                for chunk in split_message(text):
                    self.bot.send_message(chat_id, chunk)
            except Exception as e:
                logger.error(f"[{self.name}] 즉시 브리핑 실패 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "정보를 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

        @self.bot.message_handler(commands=['uptime'])
        def handle_uptime(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"서버 연속 가동 시간: {get_uptime_str()}")

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(
                chat_id,
                f"{ALARM_WELCOME}\n\n"
                "/now - 지금 바로 브리핑 받기\n"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/myid - 내 chat_id 확인\n"
                "/help - 이 도움말 보기"
            )

    def run(self):
        with self._scheduler_lock:
            if not self._scheduler_started:
                self._scheduler_started = True
                threading.Thread(target=self._scheduler_loop, daemon=True).start()
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
