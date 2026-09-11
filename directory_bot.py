import logging
from typing import Dict

import telebot
from telebot.types import Message

from common import is_allowed, split_message
from store import DirectoryStore

logger = logging.getLogger(__name__)

DIRECTORY_WELCOME = (
    "내선번호/비상연락망 너구리 가동\n\n"
    "이름, 부서, 구분 중 아무거나 그냥 입력하면 바로 검색됩니다. 명령어 없이 써도 됩니다.\n\n"
    "/find 검색어 - 검색\n"
    "/list [구분] - 전체 목록 (구분 지정 시 필터링)\n"
    "/add - 등록\n"
    "/remove 이름 - 삭제\n"
    "/help - 이 도움말"
)

ADD_USAGE = (
    "사용법: /add 구분 | 이름 | 번호 | 부서(선택) | 비고(선택)\n"
    "예시:\n"
    "/add 내선 | 홍길동 | 1234 | 총무팀\n"
    "/add 비상 | 소방서 | 119"
)


def _format_entry(e: Dict[str, str]) -> str:
    line = f"[{e.get('category', '')}] {e.get('name', '')} - {e.get('number', '')}"
    extra = " / ".join(x for x in (e.get('dept'), e.get('note')) if x)
    if extra:
        line += f" ({extra})"
    return line


def _format_entries(entries) -> str:
    entries = sorted(entries, key=lambda e: (e.get('category', ''), e.get('name', '')))
    return "\n".join(_format_entry(e) for e in entries)


class NeoguriDirectoryBot:
    def __init__(self, name: str, token: str, store: DirectoryStore):
        self.name = name
        self.bot = telebot.TeleBot(token, threaded=False)
        self.store = store
        self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=['myid'])
        def handle_myid(message: Message):
            self.bot.send_message(message.chat.id, f"chat_id: {message.chat.id}")

        @self.bot.message_handler(commands=['start', 'help'])
        def handle_start(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            self.bot.send_message(chat_id, f"{DIRECTORY_WELCOME}\n\n{ADD_USAGE}")

        @self.bot.message_handler(commands=['add'])
        def handle_add(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            raw = message.text.partition(' ')[2].strip()
            if not raw:
                self.bot.send_message(chat_id, ADD_USAGE)
                return
            fields = [f.strip() for f in raw.split('|')]
            if len(fields) < 3 or not all(fields[:3]):
                self.bot.send_message(chat_id, "구분, 이름, 번호는 필수입니다.\n\n" + ADD_USAGE)
                return
            category, name, number = fields[0], fields[1], fields[2]
            dept = fields[3] if len(fields) > 3 else ""
            note = fields[4] if len(fields) > 4 else ""
            entry = {"category": category, "name": name, "number": number, "dept": dept, "note": note}
            self.store.add(name, entry)
            logger.info(f"[{self.name}] 등록: {name} ({category})")
            self.bot.send_message(chat_id, f"등록 완료:\n{_format_entry(entry)}")

        @self.bot.message_handler(commands=['remove'])
        def handle_remove(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            name = message.text.partition(' ')[2].strip()
            if not name:
                self.bot.send_message(chat_id, "사용법: /remove 이름")
                return
            if self.store.remove(name):
                self.bot.send_message(chat_id, f"'{name}' 삭제했습니다.")
            else:
                self.bot.send_message(chat_id, f"'{name}'을(를) 찾지 못했습니다. /list로 확인하세요.")

        @self.bot.message_handler(commands=['list'])
        def handle_list(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            category_filter = message.text.partition(' ')[2].strip()
            entries = self.store.list_all()
            if category_filter:
                entries = [e for e in entries if category_filter in e.get('category', '')]
            if not entries:
                self.bot.send_message(chat_id, "등록된 항목이 없습니다. /add로 등록해주세요.")
                return
            for chunk in split_message(_format_entries(entries)):
                self.bot.send_message(chat_id, chunk)

        @self.bot.message_handler(commands=['find'])
        def handle_find(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            query = message.text.partition(' ')[2].strip()
            if not query:
                self.bot.send_message(chat_id, "사용법: /find 검색어")
                return
            self._reply_search(chat_id, query)

        @self.bot.message_handler(func=lambda m: True, content_types=['text'])
        def handle_text(message: Message):
            chat_id = message.chat.id
            if not is_allowed(chat_id):
                self.bot.send_message(chat_id, "승인된 사용자만 이용할 수 있습니다.")
                return
            self._reply_search(chat_id, message.text.strip())

    def _reply_search(self, chat_id: int, query: str):
        if not query:
            return
        results = self.store.find(query)
        if not results:
            self.bot.send_message(chat_id, f"'{query}'와(과) 일치하는 항목이 없습니다. /add로 등록하거나 /help를 확인하세요.")
            return
        for chunk in split_message(_format_entries(results)):
            self.bot.send_message(chat_id, chunk)

    def run(self):
        self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
