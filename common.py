import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer

import telebot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuietPollingConflictHandler(telebot.ExceptionHandler):
    """Render 배포 전환 중 몇 초간 이전/새 인스턴스가 동시에 getUpdates를 시도하면서
    나는 409 Conflict는 자동으로 재시도되어 해소되는 정상 노이즈다. telebot 기본 동작은
    이걸 매번 ERROR + 전체 스택트레이스로 남겨서 실제 장애를 로그에서 찾기 어렵게
    만들므로, 이 패턴만 한 줄 INFO로 조용히 남기고 나머지 예외는 그대로 telebot의
    기본 ERROR 로깅에 맡긴다."""

    def handle(self, exception) -> bool:
        msg = str(exception)
        if "terminated by other getUpdates request" in msg or "Conflict" in msg:
            logger.info(f"배포 전환 중 폴링 충돌(정상, 자동 재시도): {msg}")
            return True
        return False


def make_telebot(token: str) -> telebot.TeleBot:
    return telebot.TeleBot(token, threaded=False, exception_handler=QuietPollingConflictHandler())

TELEGRAM_MAX_LEN = 4000
START_TIME = time.time()
KST = timezone(timedelta(hours=9))

# run_forever()가 갱신하는 봇별 실제 상태. /health가 정적 "OK" 대신 이걸 그대로 보여준다.
_bot_status_lock = threading.Lock()
BOT_STATUS: Dict[str, dict] = {}


def _set_bot_status(name: str, **fields):
    with _bot_status_lock:
        BOT_STATUS.setdefault(name, {})
        BOT_STATUS[name].update(fields)
        BOT_STATUS[name]["updated_at"] = time.time()


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _bot_status_lock:
            snapshot = {name: dict(info) for name, info in BOT_STATUS.items()}
        # 배포 검증(Render 헬스체크)이 이 엔드포인트로 죽지 않도록 HTTP 상태는 항상 200으로 두고,
        # 실제 상태는 바디에 담아 "정상"이라는 거짓말 대신 진짜 정보를 보여준다.
        payload = {
            "status": "ok" if snapshot and all(b.get("state") == "running" for b in snapshot.values()) else "starting_or_degraded",
            "bots": snapshot,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), HealthCheckHandler).serve_forever()


def _parse_allowed_ids(raw: Optional[str]) -> set:
    if not raw:
        return set()
    return {int(t.strip()) for t in raw.split(",") if t.strip().lstrip("-").isdigit()}


ALLOWED_CHAT_IDS = _parse_allowed_ids(os.environ.get("ALLOWED_CHAT_IDS"))

if not ALLOWED_CHAT_IDS:
    logger.warning(
        "ALLOWED_CHAT_IDS가 설정되지 않아 모든 명령이 차단됩니다. "
        "/myid로 본인의 chat_id를 확인한 뒤 환경변수에 등록하세요."
    )


def is_allowed(chat_id: int) -> bool:
    """ALLOWED_CHAT_IDS가 비어 있으면 fail-closed(전체 차단)로 동작한다.
    /myid만은 부트스트랩을 위해 이 검사를 우회해서 항상 응답한다."""
    return chat_id in ALLOWED_CHAT_IDS


def get_uptime_str() -> str:
    elapsed = int(time.time() - START_TIME)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}시간 {minutes}분 {seconds}초"


def _extract_token_usage(response) -> Optional[tuple]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is None:
        return None
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    return prompt_tokens, output_tokens, total_tokens


def format_token_usage(response) -> str:
    usage = _extract_token_usage(response)
    if usage is None:
        return ""
    prompt_tokens, output_tokens, total_tokens = usage
    return f"\n\n토큰 사용: 입력 {prompt_tokens:,} · 출력 {output_tokens:,} · 합계 {total_tokens:,}"


_usage_store = None  # main.py가 set_usage_store()로 한 번 주입한다.


def set_usage_store(store) -> None:
    global _usage_store
    _usage_store = store


def log_token_usage(name: str, response) -> None:
    """호출마다 실제 토큰 사용량을 서버 로그에 남기고(Render 로그에서 '토큰 사용:'으로 필터링 가능),
    usage store가 설정돼 있으면 날짜별 누적치도 함께 쌓는다. 지금까지는 이 기록이 전혀 없었다."""
    usage = _extract_token_usage(response)
    if usage is None:
        return
    prompt_tokens, output_tokens, total_tokens = usage
    logger.info(f"[{name}] 토큰 사용: 입력 {prompt_tokens:,} · 출력 {output_tokens:,} · 합계 {total_tokens:,}")
    if _usage_store is not None:
        try:
            date_str = datetime.now(KST).strftime("%Y-%m-%d")
            _usage_store.add(date_str, name, total_tokens)
        except Exception as e:
            logger.warning(f"[{name}] 사용량 집계 저장 실패: {e}")


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


class TelegramBotBase:
    """모든 너구리 봇이 공통으로 쓰는 /myid, /uptime, /usage 명령과 사용자 인증 가드,
    AI 호출 쿨다운. 상속하는 쪽에서 self.bot(TeleBot 인스턴스)을 먼저 만들어둔 뒤 써야 한다."""

    AI_COOLDOWN_SECONDS = 1.5

    def _guard(self, chat_id: int) -> bool:
        """허용된 사용자면 True, 아니면 안내 메시지를 보내고 False를 반환한다."""
        if is_allowed(chat_id):
            return True
        self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
        return False

    def _ai_cooldown_ok(self, chat_id: int) -> bool:
        """같은 사용자가 아주 짧은 간격으로 AI 호출을 반복하면(실수로 인한 폭주 등)
        API 호출이 그대로 새나가지 않도록 잠깐 막는다. 정상적인 대화 속도에는 영향 없음."""
        if not hasattr(self, "_ai_last_call_at"):
            self._ai_last_call_at = {}
        now = time.time()
        last = self._ai_last_call_at.get(chat_id, 0)
        if now - last < self.AI_COOLDOWN_SECONDS:
            return False
        self._ai_last_call_at[chat_id] = now
        return True

    def _register_common_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message):
            # ALLOWED_CHAT_IDS 등록 전에도 본인 chat_id를 확인할 수 있어야 하므로
            # 이 명령만 인증 검사를 우회한다.
            self.bot.send_message(message.chat.id, f"chat_id: {message.chat.id}")

        @self.bot.message_handler(commands=['uptime'])
        def handle_uptime(message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(chat_id, f"서버 연속 가동 시간: {get_uptime_str()}")

        @self.bot.message_handler(commands=['usage'])
        def handle_usage(message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if _usage_store is None:
                self.bot.send_message(chat_id, "사용량 집계 기능이 설정되지 않았습니다.")
                return
            today = datetime.now(KST).date()
            today_str = today.strftime("%Y-%m-%d")
            week_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
            try:
                today_usage = _usage_store.get_day(today_str)
                week_usage = _usage_store.get_range(week_dates)
            except Exception as e:
                logger.warning(f"사용량 조회 실패: {e}")
                self.bot.send_message(chat_id, "사용량 조회 중 오류가 발생했습니다.")
                return

            def _fmt(usage: Dict[str, int]) -> str:
                if not usage:
                    return "(기록 없음)"
                lines = [f"- {n}: {c:,} 토큰" for n, c in sorted(usage.items(), key=lambda kv: -kv[1])]
                lines.append(f"합계: {sum(usage.values()):,} 토큰")
                return "\n".join(lines)

            self.bot.send_message(
                chat_id,
                f"[오늘 {today_str}]\n{_fmt(today_usage)}\n\n[최근 7일 합계]\n{_fmt(week_usage)}"
            )


def run_forever(bot_obj, name: str):
    while True:
        try:
            logger.info(f"[{name}] 폴링 시작")
            _set_bot_status(name, state="running", error=None, started_at=time.time())
            bot_obj.run()
            # infinity_polling은 정상 상황에서 반환되지 않지만, 혹시 반환되면 상태를 남긴다.
            _set_bot_status(name, state="stopped")
        except Exception as e:
            logger.error(f"[{name}] 폴링이 예외로 중단됨, 5초 후 재시작: {e}", exc_info=True)
            _set_bot_status(name, state="crashed", error=str(e))
            time.sleep(5)
