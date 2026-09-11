import os
import time
import logging
from typing import List, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4000
START_TIME = time.time()


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


def format_token_usage(response) -> str:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return ""
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is None:
        return ""
    return f"\n\n토큰 사용: 입력 {prompt_tokens:,} · 출력 {output_tokens:,} · 합계 {total_tokens:,}"


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
    """모든 너구리 봇이 공통으로 쓰는 /myid, /uptime 명령과 사용자 인증 가드.
    상속하는 쪽에서 self.bot(TeleBot 인스턴스)을 먼저 만들어둔 뒤 써야 한다."""

    def _guard(self, chat_id: int) -> bool:
        """허용된 사용자면 True, 아니면 안내 메시지를 보내고 False를 반환한다."""
        if is_allowed(chat_id):
            return True
        self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
        return False

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


def run_forever(bot_obj, name: str):
    while True:
        try:
            logger.info(f"[{name}] 폴링 시작")
            bot_obj.run()
        except Exception as e:
            logger.error(f"[{name}] 폴링이 예외로 중단됨, 5초 후 재시작: {e}", exc_info=True)
            time.sleep(5)
