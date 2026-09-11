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

# Gemini Flash 계열 추정 단가(변경 가능) — 실제 청구는 ai.google.dev/pricing 기준
GEMINI_INPUT_USD_PER_1M = float(os.environ.get("GEMINI_INPUT_USD_PER_1M", "0.1"))
GEMINI_OUTPUT_USD_PER_1M = float(os.environ.get("GEMINI_OUTPUT_USD_PER_1M", "0.4"))
USD_TO_KRW = float(os.environ.get("USD_TO_KRW", "1400"))


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


def estimate_cost_krw(prompt_tokens: int, output_tokens: int) -> float:
    usd = (prompt_tokens * GEMINI_INPUT_USD_PER_1M + output_tokens * GEMINI_OUTPUT_USD_PER_1M) / 1_000_000
    return usd * USD_TO_KRW


def format_token_usage(response) -> str:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return ""
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is None:
        return ""
    krw = estimate_cost_krw(prompt_tokens, output_tokens)
    cost_str = "1원 미만" if krw < 1 else f"약 {krw:,.0f}원"
    return (
        f"\n\n🔢 토큰 사용: 입력 {prompt_tokens:,} · 출력 {output_tokens:,} · 합계 {total_tokens:,}"
        f"\n💰 예상 비용: {cost_str} (Gemini Flash 계열 추정 단가 기준, 실제 단가는 ai.google.dev/pricing 참고)"
    )


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
