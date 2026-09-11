import time
import logging
import threading
from datetime import datetime
import telebot
from telebot.types import Message
from google.genai import types

from common import split_message, log_token_usage, ALLOWED_CHAT_IDS, TelegramBotBase, KST
from router import GeminiRouter
from store import AlarmScheduleStore

logger = logging.getLogger(__name__)

ALARM_CHECK_INTERVAL_SECONDS = 60  # 정시가 됐는지 이 주기로 확인 (실제 발송은 시간대당 1회로 제한됨)

ALARM_INSTRUCTION = (
    "너는 시장/뉴스 브리핑 AI다. 반드시 구글 검색을 실제로 호출해 방금 확인한 정보만으로 "
    "아래 항목을 채워라. 검색 없이 추측한 수치는 절대 금지다.\n\n"
    "매우 중요: 오늘 날짜는 네 스스로 추측하지 말고, 사용자 메시지에 명시된 날짜만 사실로 여겨라. "
    "검색어를 만들 때도 반드시 그 날짜(또는 '오늘', '현재' 같은 표현)를 그대로 써라. 그 날짜와 다른 "
    "과거 날짜의 시세·뉴스를 가져오면 안 된다.\n\n"
    "[시장 지표] 아래 6개 지표를 현재가·전일 대비 등락·등락률과 함께 적고, 지표마다 등락 배경을 "
    "한 줄로 짧게 덧붙여라(예: '반도체주 강세에 상승', 'FOMC 발언 경계감에 하락'):\n"
    "1. 코스피\n2. 나스닥\n3. 원/달러 환율\n4. 달러인덱스(DXY)\n5. 국제유가(WTI)\n6. 비트코인\n\n"
    "[주요 뉴스] 경제·사회·국제 등 서로 다른 분야를 섞어서 헤드라인 7~8개를 고르고, 각각 한 줄로 "
    "요약하라. 같은 주제만 몰아서 고르지 마라.\n\n"
    "확인 못한 항목은 '확인 불가'라고 써라. 인사말이나 군더더기 설명 없이 위 형식만 채워라. "
    "마크다운 특수기호와 이모지는 쓰지 말고, 통화 기호(₩, $)만 예외로 허용한다."
)


def _build_briefing_prompt() -> str:
    now = datetime.now(KST)
    date_str = now.strftime("%Y년 %m월 %d일")
    time_str = now.strftime("%H시 %M분")
    return (
        f"지금은 한국 시간 기준 {date_str} {time_str}이다. 이 날짜를 기준으로 가장 최근/현재 "
        "코스피·나스닥·원달러환율·달러인덱스·국제유가·비트코인 시세와, 경제·사회·국제 등 다양한 "
        "분야의 주요 뉴스 헤드라인을 정리해줘. 검색어에도 이 날짜를 반영해서, "
        "절대 다른 날짜의 옛날 데이터를 가져오지 마라."
    )

ALARM_WELCOME = (
    "알람너구리 가동\n\n"
    "한국 시간(KST) 기준 매시 정각마다 자동으로 코스피·나스닥·원달러환율·달러인덱스·국제유가·"
    "비트코인 시세와 분야별 주요 뉴스를 정리해서 보내드립니다 (시간당 최대 1회).\n"
    "지금 바로 확인하고 싶으면 /now를 입력하세요."
)


class NeoguriAlarmBot(TelegramBotBase):
    def __init__(self, name: str, token: str, router: GeminiRouter, schedule_store: AlarmScheduleStore,
                 check_interval_seconds: int = ALARM_CHECK_INTERVAL_SECONDS):
        self.name = name
        self.bot = telebot.TeleBot(token, threaded=False)
        self.router = router
        self.schedule_store = schedule_store
        self.check_interval_seconds = check_interval_seconds
        self.config = types.GenerateContentConfig(
            system_instruction=ALARM_INSTRUCTION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,  # 시세·수치 브리핑이라 창의성보다 일관성이 중요
            # 지표 6개 + 헤드라인 7~8개짜리 긴 브리핑인 데다, 검색 도구가 항상 켜져 있어 응답 전에
            # 모델이 검색 계획을 세우는 데도 토큰을 쓴다. 너무 타이트하면 실제 답변이 나오기 전에
            # 한도에 걸려 문장이 중간에 끊길 수 있어 여유 있게 잡는다.
            max_output_tokens=3000,
        )
        self._scheduler_started = False
        self._scheduler_lock = threading.Lock()
        self._register_handlers()

    def _build_briefing(self) -> str:
        """가끔 모델이 지시를 어기고 검색 도구를 안 부른 채 답하는 경우가 있다(관찰상 10회 중
        1회 정도). 그러면 ALARM_INSTRUCTION대로 전부 '확인 불가'만 나오는 빈 브리핑이 되므로,
        검색 근거가 없으면 최대 2번까지 다시 시도한다."""
        response = None
        for attempt in range(3):
            response = self.router.generate(contents=_build_briefing_prompt(), config=self.config)
            log_token_usage(self.name, response)
            if self._log_grounding_info(response):
                break
            if attempt < 2:
                logger.warning(f"[{self.name}] 검색 없이 응답이 와서 재시도합니다 ({attempt + 1}/3)")
        return response.text or "정보를 가져오지 못했습니다."

    def _log_grounding_info(self, response) -> bool:
        """실제로 구글 검색을 근거로 답했는지, 어떤 검색어/출처를 썼는지 로그로 남긴다.
        브리핑 내용이 부정확하다는 의심이 들 때 이 로그로 원인(검색 미실행 vs 검색은 했지만
        결과 해석 오류)을 구분할 수 있다. 반환값은 이번 응답이 검색 근거를 갖고 있는지 여부."""
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                logger.warning(f"[{self.name}] 응답에 candidate가 없어 검색 근거를 확인할 수 없습니다 (모델: {self.router.model_name})")
                return False
            metadata = getattr(candidates[0], "grounding_metadata", None)
            if not metadata:
                logger.warning(
                    f"[{self.name}] 이번 응답은 구글 검색 없이 생성됨(grounding_metadata 없음) — "
                    f"모델이 검색 도구를 호출하지 않고 답했을 가능성이 높습니다 (모델: {self.router.model_name})"
                )
                return False
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
            return True
        except Exception as e:
            logger.warning(f"[{self.name}] 검색 근거 로깅 실패: {e}")
            return False

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

    def _current_hour_slot(self) -> str:
        return datetime.now(KST).strftime("%Y-%m-%d %H")

    def _scheduler_loop(self):
        while True:
            try:
                current_slot = self._current_hour_slot()
                try:
                    last_slot = self.schedule_store.get_last_sent_hour()
                except Exception as e:
                    logger.warning(f"[{self.name}] 마지막 발송 시각 조회 실패, 이번 확인은 건너뜁니다: {e}")
                    last_slot = current_slot  # 조회 실패 시 중복 발송 대신 이번 턴은 건너뛴다
                if current_slot != last_slot:
                    self._broadcast()
                    try:
                        self.schedule_store.set_last_sent_hour(current_slot)
                    except Exception as e:
                        logger.warning(f"[{self.name}] 마지막 발송 시각 저장 실패: {e}")
            except Exception as e:
                logger.error(f"[{self.name}] 정기 브리핑 실패: {e}", exc_info=True)
            time.sleep(self.check_interval_seconds)

    def _register_handlers(self):
        self._register_common_handlers()

        @self.bot.message_handler(commands=['start'])
        def handle_start(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(chat_id, ALARM_WELCOME)

        @self.bot.message_handler(commands=['now'])
        def handle_now(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self._ai_cooldown_ok(chat_id):
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                text = self._build_briefing()
                for chunk in split_message(text):
                    self.bot.send_message(chat_id, chunk)
            except Exception as e:
                logger.error(f"[{self.name}] 즉시 브리핑 실패 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "정보를 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

        @self.bot.message_handler(commands=['help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(
                chat_id,
                f"{ALARM_WELCOME}\n\n"
                "/now - 지금 바로 브리핑 받기\n"
                "/uptime - 서버 연속 가동 시간 확인\n"
                "/usage - 오늘/최근 7일 토큰 사용량 확인 (전체 봇 합산)\n"
                "/myid - 내 chat_id 확인\n"
                "/help - 이 도움말 보기"
            )

    def run(self):
        with self._scheduler_lock:
            if not self._scheduler_started:
                self._scheduler_started = True
                threading.Thread(target=self._scheduler_loop, daemon=True).start()
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
