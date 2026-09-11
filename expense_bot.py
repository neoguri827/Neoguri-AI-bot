import json
import logging
from datetime import datetime, timedelta

import telebot
from telebot.types import Message
from google.genai import types

from common import TelegramBotBase, log_token_usage, KST
from router import GeminiRouter
from store import ExpenseStore

logger = logging.getLogger(__name__)

EXPENSE_CATEGORIES = ["식비", "카페", "교통", "쇼핑", "의료", "생활", "여가", "기타"]

EXPENSE_INSTRUCTION = (
    "너는 지출 내역을 구조화된 데이터로 추출하는 엔진이다. 사용자가 보낸 영수증 사진이나 "
    "텍스트(예: '스타벅스 아메리카노 4500원')에서 지출 정보를 뽑아라.\n\n"
    "금액은 반드시 숫자(원 단위 정수)만 추출하고, 콤마·통화기호는 빼라. "
    "상호명이 안 보이면 빈 문자열로 둬라. 카테고리는 반드시 다음 중 하나로 골라라: "
    f"{', '.join(EXPENSE_CATEGORIES)}.\n"
    "영수증에 날짜가 보이면 YYYY-MM-DD 형식으로 적고, 안 보이면 빈 문자열로 둬라.\n"
    "지출 내역이 아니거나 금액을 알 수 없으면 success를 false로 하고 note에 이유를 짧게 적어라. "
    "추측해서 숫자를 지어내지 마라."
)

EXPENSE_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "merchant": {"type": "string"},
        "amount": {"type": "integer"},
        "category": {"type": "string", "enum": EXPENSE_CATEGORIES},
        "date": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["success", "merchant", "amount", "category", "date", "note"],
}

EXPENSE_WELCOME = (
    "가계부 너구리 가동\n\n"
    "영수증 사진을 보내거나 \"스타벅스 아메리카노 4500원\"처럼 글로 적으면 자동으로 기록합니다.\n\n"
    "/week - 이번 주 지출 요약\n"
    "/month - 이번 달 지출 요약\n"
    "/list - 최근 지출 목록 (삭제용 번호 포함)\n"
    "/delete 번호 - 해당 지출 삭제\n"
    "/reset - 전체 기록 삭제\n"
    "/help - 이 도움말 보기"
)

LIST_DISPLAY_LIMIT = 20


class NeoguriExpenseBot(TelegramBotBase):
    def __init__(self, name: str, token: str, router: GeminiRouter, store: ExpenseStore):
        self.name = name
        self.bot = telebot.TeleBot(token, threaded=False)
        self.router = router
        self.store = store
        self.config = types.GenerateContentConfig(
            system_instruction=EXPENSE_INSTRUCTION,
            temperature=0.1,  # 구조화 추출이라 일관성이 중요
            max_output_tokens=500,
            response_mime_type="application/json",
            response_schema=EXPENSE_SCHEMA,
        )
        self._register_handlers()

    def _extract(self, contents) -> dict:
        response = self.router.generate(contents=contents, config=self.config)
        log_token_usage(self.name, response)
        return json.loads(response.text)

    def _record_and_confirm(self, chat_id: int, parsed: dict):
        if not parsed.get("success"):
            reason = parsed.get("note") or "지출 정보를 확인하지 못했습니다."
            self.bot.send_message(chat_id, f"기록하지 못했어요: {reason}")
            return
        amount = parsed.get("amount") or 0
        if amount <= 0:
            self.bot.send_message(chat_id, "금액을 확인하지 못했어요. 다시 한번 보내주세요.")
            return
        record = {
            "merchant": parsed.get("merchant") or "",
            "amount": amount,
            "category": parsed.get("category") or "기타",
            "receipt_date": parsed.get("date") or "",
            "logged_date": datetime.now(KST).strftime("%Y-%m-%d"),
        }
        entry_id = self.store.add(chat_id, record)
        merchant_part = f"{record['merchant']} · " if record["merchant"] else ""
        self.bot.send_message(
            chat_id,
            f"기록했습니다 (#{entry_id})\n{merchant_part}{record['category']} · {amount:,}원"
        )

    def _week_start(self) -> str:
        today = datetime.now(KST).date()
        monday = today - timedelta(days=today.weekday())
        return monday.strftime("%Y-%m-%d")

    def _month_start(self) -> str:
        return datetime.now(KST).date().replace(day=1).strftime("%Y-%m-%d")

    def _summarize_since(self, chat_id: int, since_date: str) -> dict:
        totals: dict = {}
        for entry in self.store.list_all(chat_id):
            logged = entry.get("logged_date", "")
            if logged >= since_date:
                category = entry.get("category", "기타")
                totals[category] = totals.get(category, 0) + int(entry.get("amount", 0))
        return totals

    def _send_summary(self, chat_id: int, since_date: str, title: str):
        totals = self._summarize_since(chat_id, since_date)
        if not totals:
            self.bot.send_message(chat_id, f"{title} 지출 기록이 없습니다.")
            return
        lines = [f"- {cat}: {amt:,}원" for cat, amt in sorted(totals.items(), key=lambda kv: -kv[1])]
        lines.append(f"합계: {sum(totals.values()):,}원")
        self.bot.send_message(chat_id, f"[{title}]\n" + "\n".join(lines))

    def _register_handlers(self):
        self._register_common_handlers()

        @self.bot.message_handler(commands=['start', 'help'])
        def handle_help(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.bot.send_message(chat_id, EXPENSE_WELCOME)

        @self.bot.message_handler(commands=['week'])
        def handle_week(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self._send_summary(chat_id, self._week_start(), "이번 주")

        @self.bot.message_handler(commands=['month'])
        def handle_month(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self._send_summary(chat_id, self._month_start(), "이번 달")

        @self.bot.message_handler(commands=['list'])
        def handle_list(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            entries = self.store.list_all(chat_id)
            if not entries:
                self.bot.send_message(chat_id, "기록된 지출이 없습니다.")
                return
            entries.sort(key=lambda e: e.get("logged_date", ""), reverse=True)
            recent = entries[:LIST_DISPLAY_LIMIT]
            lines = []
            for e in recent:
                merchant = f"{e['merchant']} · " if e.get("merchant") else ""
                lines.append(
                    f"#{e['id']} {e.get('logged_date', '')} {merchant}"
                    f"{e.get('category', '기타')} {int(e.get('amount', 0)):,}원"
                )
            extra = f"\n...외 {len(entries) - LIST_DISPLAY_LIMIT}건" if len(entries) > LIST_DISPLAY_LIMIT else ""
            self.bot.send_message(chat_id, "\n".join(lines) + extra + "\n\n/delete 번호 로 삭제할 수 있습니다.")

        @self.bot.message_handler(commands=['delete'])
        def handle_delete(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            entry_id = message.text.partition(' ')[2].strip()
            if not entry_id:
                self.bot.send_message(chat_id, "사용법: /delete 번호 (번호는 /list로 확인)")
                return
            if self.store.remove(chat_id, entry_id):
                self.bot.send_message(chat_id, f"#{entry_id} 삭제했습니다.")
            else:
                self.bot.send_message(chat_id, f"#{entry_id}를 찾지 못했습니다. /list로 확인하세요.")

        @self.bot.message_handler(commands=['reset'])
        def handle_reset(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            self.store.clear(chat_id)
            self.bot.send_message(chat_id, "전체 지출 기록을 삭제했습니다.")

        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self._ai_cooldown_ok(chat_id):
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                file_id = message.photo[-1].file_id
                file_info = self.bot.get_file(file_id)
                file_bytes = self.bot.download_file(file_info.file_path)
                parsed = self._extract([
                    types.Part.from_bytes(data=file_bytes, mime_type="image/jpeg"),
                    "이 영수증 사진에서 지출 정보를 추출해줘.",
                ])
                self._record_and_confirm(chat_id, parsed)
            except Exception as e:
                logger.error(f"[{self.name}] 영수증 처리 실패 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "영수증을 처리하는 중 오류가 발생했습니다.")

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not self._guard(chat_id):
                return
            if not self._ai_cooldown_ok(chat_id):
                return
            self.bot.send_chat_action(chat_id, 'typing')
            try:
                parsed = self._extract(f"지출 내역: {message.text}")
                self._record_and_confirm(chat_id, parsed)
            except Exception as e:
                logger.error(f"[{self.name}] 지출 처리 실패 (Chat ID: {chat_id}): {e}", exc_info=True)
                self.bot.send_message(chat_id, "처리하는 중 오류가 발생했습니다.")

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
