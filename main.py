import os
import threading

from google import genai

from common import start_health_server, run_forever, logger
from router import GeminiRouter, resolve_api_key
from store import ChatHistoryStore
from translator_bot import NeoguriTranslatorBot, TRANSLATOR_BOT_DEFS
from memory_bot import MemoryGeminiBot
from assistant_bot import ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME
from mail_bot import MAIL_INSTRUCTION, MAIL_WELCOME

threading.Thread(target=start_health_server, daemon=True).start()

translate_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_TRANSLATE")), label="번역")
assistant_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_ASSISTANT")), label="비서")
mail_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_MAIL")), label="메일")

UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN 환경변수가 필요합니다.")


def main():
    threads = []

    for cfg in TRANSLATOR_BOT_DEFS:
        token = os.environ.get(cfg["token_env"])
        if not token:
            logger.warning(f"{cfg['token_env']} 없어서 {cfg['name']} 건너뜁니다.")
            continue
        bot_obj = NeoguriTranslatorBot(cfg["name"], token, cfg["instruction"], translate_router)
        t = threading.Thread(target=run_forever, args=(bot_obj, cfg["name"]), daemon=True)
        t.start()
        threads.append(t)

    smart_token = os.environ.get("TELEGRAM_TOKEN_SMART")
    if smart_token:
        store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="assistant")
        bot_obj = MemoryGeminiBot("똑똑한 너구리", smart_token, assistant_router, store,
                                   ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME)
        t = threading.Thread(target=run_forever, args=(bot_obj, "똑똑한 너구리"), daemon=True)
        t.start()
        threads.append(t)
    else:
        logger.warning("TELEGRAM_TOKEN_SMART 없어서 똑똑한 너구리는 건너뜁니다.")

    mail_token = os.environ.get("TELEGRAM_TOKEN_MAIL")
    if mail_token:
        store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="mail")
        bot_obj = MemoryGeminiBot("메일작성용 너구리", mail_token, mail_router, store,
                                   MAIL_INSTRUCTION, MAIL_WELCOME)
        t = threading.Thread(target=run_forever, args=(bot_obj, "메일작성용 너구리"), daemon=True)
        t.start()
        threads.append(t)
    else:
        logger.warning("TELEGRAM_TOKEN_MAIL 없어서 메일작성용 너구리는 건너뜁니다.")

    if not threads:
        raise ValueError("실행 가능한 봇이 없습니다. 환경변수를 확인하세요.")

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
