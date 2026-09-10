import os
import threading

from google import genai

from common import start_health_server, run_forever, logger, ALLOWED_CHAT_IDS
from router import GeminiRouter, resolve_api_key
from store import ChatHistoryStore, KnowledgeStore
from translator_bot import NeoguriTranslatorBot, TRANSLATOR_BOT_DEFS
from memory_bot import MemoryGeminiBot
from assistant_bot import ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME, ASSISTANT_QUICK_COMMANDS
from mail_bot import MAIL_INSTRUCTION, MAIL_WELCOME

threading.Thread(target=start_health_server, daemon=True).start()

UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN 환경변수가 필요합니다.")


def main():
    threads = []
    bot_registry = []

    translate_router = None
    translator_tokens = {cfg["name"]: os.environ.get(cfg["token_env"]) for cfg in TRANSLATOR_BOT_DEFS}
    if any(translator_tokens.values()):
        translate_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_TRANSLATE")), label="번역")
        for cfg in TRANSLATOR_BOT_DEFS:
            token = translator_tokens[cfg["name"]]
            if not token:
                logger.warning(f"{cfg['token_env']} 없어서 {cfg['name']} 건너뜁니다.")
                continue
            bot_obj = NeoguriTranslatorBot(cfg["name"], token, cfg["instruction"], translate_router)
            bot_registry.append(bot_obj)
            t = threading.Thread(target=run_forever, args=(bot_obj, cfg["name"]), daemon=True)
            t.start()
            threads.append(t)

    assistant_router = None
    smart_token = os.environ.get("TELEGRAM_TOKEN_SMART")
    if smart_token:
        assistant_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_ASSISTANT")), label="비서")
        store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="assistant")
        kb_store = KnowledgeStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="assistant")
        bot_obj = MemoryGeminiBot("똑똑한 너구리", smart_token, assistant_router, store,
                                   ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME,
                                   quick_commands=ASSISTANT_QUICK_COMMANDS,
                                   knowledge_store=kb_store)
        bot_registry.append(bot_obj)
        t = threading.Thread(target=run_forever, args=(bot_obj, "똑똑한 너구리"), daemon=True)
        t.start()
        threads.append(t)
    else:
        logger.warning("TELEGRAM_TOKEN_SMART 없어서 똑똑한 너구리는 건너뜁니다.")

    mail_router = None
    mail_token = os.environ.get("TELEGRAM_TOKEN_MAIL")
    if mail_token:
        mail_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_MAIL")), label="메일")
        store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="mail")
        bot_obj = MemoryGeminiBot("메일작성용 너구리", mail_token, mail_router, store,
                                   MAIL_INSTRUCTION, MAIL_WELCOME)
        bot_registry.append(bot_obj)
        t = threading.Thread(target=run_forever, args=(bot_obj, "메일작성용 너구리"), daemon=True)
        t.start()
        threads.append(t)
    else:
        logger.warning("TELEGRAM_TOKEN_MAIL 없어서 메일작성용 너구리는 건너뜁니다.")

    if not threads:
        raise ValueError("실행 가능한 봇이 없습니다. 환경변수를 확인하세요.")

    notifier_bot = next((b for b in bot_registry if getattr(b, "name", "") == "똑똑한 너구리"), None)
    if notifier_bot is None and bot_registry:
        notifier_bot = bot_registry[0]

    if notifier_bot is not None:
        def notify_downgrade(label: str, old_model: str, new_model: str):
            text = f"⚠️ [{label}] 모델이 하위 등급으로 전환됐습니다.\n{old_model} → {new_model}"
            if not ALLOWED_CHAT_IDS:
                logger.warning(f"알림 받을 chat_id가 없습니다(ALLOWED_CHAT_IDS 미설정): {text}")
                return
            for chat_id in ALLOWED_CHAT_IDS:
                try:
                    notifier_bot.bot.send_message(chat_id, text)
                except Exception as e:
                    logger.warning(f"다운그레이드 알림 전송 실패(chat_id={chat_id}): {e}")

        for router in (translate_router, assistant_router, mail_router):
            if router is not None:
                router.on_downgrade = notify_downgrade

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
