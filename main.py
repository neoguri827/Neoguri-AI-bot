import os
import threading

from google import genai

from common import start_health_server, run_forever, logger
from router import GeminiRouter, resolve_api_key
from store import ChatHistoryStore, KnowledgeStore
from translator_bot import NeoguriTranslatorBot, TRANSLATOR_BOT_DEFS
from memory_bot import MemoryGeminiBot
from assistant_bot import ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME, ASSISTANT_QUICK_COMMANDS
from mail_bot import MAIL_INSTRUCTION, MAIL_WELCOME
from puppy_bot import PUPPY_INSTRUCTION, PUPPY_WELCOME
from alarm_bot import NeoguriAlarmBot

threading.Thread(target=start_health_server, daemon=True).start()


def _get_env_token(name: str):
    """환경변수에 실수로 섞여 들어간 공백/줄바꿈 때문에 텔레그램 토큰 검증이
    실패해서 프로세스 전체가 죽는 일을 막기 위해 앞뒤 공백을 제거한다."""
    value = os.environ.get(name)
    return value.strip() if value else value


UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise ValueError("UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN 환경변수가 필요합니다.")


def main():
    threads = []

    translator_tokens = {cfg["name"]: _get_env_token(cfg["token_env"]) for cfg in TRANSLATOR_BOT_DEFS}
    if any(translator_tokens.values()):
        translate_router = None
        try:
            translate_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_TRANSLATE")), label="번역")
        except Exception as e:
            logger.error(f"번역 라우터 초기화 실패, 번역봇 전체를 건너뜁니다: {e}", exc_info=True)
        if translate_router is not None:
            for cfg in TRANSLATOR_BOT_DEFS:
                token = translator_tokens[cfg["name"]]
                if not token:
                    logger.warning(f"{cfg['token_env']} 없어서 {cfg['name']} 건너뜁니다.")
                    continue
                try:
                    bot_obj = NeoguriTranslatorBot(cfg["name"], token, cfg["instruction"], translate_router)
                except Exception as e:
                    logger.error(f"{cfg['name']} 초기화 실패, 건너뜁니다: {e}", exc_info=True)
                    continue
                t = threading.Thread(target=run_forever, args=(bot_obj, cfg["name"]), daemon=True)
                t.start()
                threads.append(t)

    smart_token = _get_env_token("TELEGRAM_TOKEN_SMART")
    if smart_token:
        try:
            assistant_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_ASSISTANT")), label="비서")
            store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="assistant")
            kb_store = KnowledgeStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="assistant")
            bot_obj = MemoryGeminiBot("똑똑한 너구리", smart_token, assistant_router, store,
                                       ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME,
                                       quick_commands=ASSISTANT_QUICK_COMMANDS,
                                       knowledge_store=kb_store,
                                       enable_token_usage=True,
                                       enable_session_confirmation=True)
            t = threading.Thread(target=run_forever, args=(bot_obj, "똑똑한 너구리"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"똑똑한 너구리 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_SMART 없어서 똑똑한 너구리는 건너뜁니다.")

    mail_token = _get_env_token("TELEGRAM_TOKEN_MAIL")
    if mail_token:
        try:
            mail_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_MAIL")), label="메일")
            store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="mail")
            bot_obj = MemoryGeminiBot("메일작성용 너구리", mail_token, mail_router, store,
                                       MAIL_INSTRUCTION, MAIL_WELCOME)
            t = threading.Thread(target=run_forever, args=(bot_obj, "메일작성용 너구리"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"메일작성용 너구리 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_MAIL 없어서 메일작성용 너구리는 건너뜁니다.")

    puppy_token = _get_env_token("TELEGRAM_TOKEN_CASUAL")
    if puppy_token:
        try:
            puppy_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_CASUAL")), label="개아")
            store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="casual")
            bot_obj = MemoryGeminiBot("개아", puppy_token, puppy_router, store,
                                       PUPPY_INSTRUCTION, PUPPY_WELCOME)
            t = threading.Thread(target=run_forever, args=(bot_obj, "개아"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"개아 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_CASUAL 없어서 개아는 건너뜁니다.")

    alarm_token = _get_env_token("TELEGRAM_TOKEN_ALARM")
    if alarm_token:
        try:
            alarm_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_ALARM")), label="알람")
            bot_obj = NeoguriAlarmBot("알람너구리", alarm_token, alarm_router)
            t = threading.Thread(target=run_forever, args=(bot_obj, "알람너구리"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"알람너구리 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_ALARM 없어서 알람너구리는 건너뜁니다.")

    if not threads:
        raise ValueError("실행 가능한 봇이 없습니다. 환경변수를 확인하세요.")

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
