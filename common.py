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


def is_allowed(chat_id: int) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    return chat_id in ALLOWED_CHAT_IDS


def get_uptime_str() -> str:
    elapsed = int(time.time() - START_TIME)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}시간 {minutes}분 {seconds}초"


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


def run_forever(bot_obj, name: str):
    while True:
        try:
            logger.info(f"[{name}] 폴링 시작")
            bot_obj.run()
        except Exception as e:
            logger.error(f"[{name}] 폴링이 예외로 중단됨, 5초 후 재시작: {e}", exc_info=True)
            time.sleep(5)
