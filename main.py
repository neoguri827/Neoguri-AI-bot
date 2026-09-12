import os
import threading

from google import genai

from common import start_health_server, run_forever, set_usage_store, logger
from router import GeminiRouter, resolve_api_key
from store import ChatHistoryStore, KnowledgeStore, UsageStore
from translator_bot import NeoguriTranslatorBot, TRANSLATOR_BOT_DEFS
from memory_bot import MemoryGeminiBot
from assistant_bot import ASSISTANT_INSTRUCTION, ASSISTANT_WELCOME, ASSISTANT_QUICK_COMMANDS, classify_complexity
from mail_bot import MAIL_INSTRUCTION, MAIL_WELCOME
from puppy_bot import PUPPY_INSTRUCTION, PUPPY_WELCOME
from directory_bot import DIRECTORY_INSTRUCTION, DIRECTORY_WELCOME
from document_bot import NeoguriDocumentBot

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

    set_usage_store(UsageStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="usage"))

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
                                       enable_session_confirmation=True,
                                       complexity_classifier=classify_complexity,
                                       temperature={"flash": 0.5, "pro": 0.2},
                                       max_output_tokens={"flash": 1536, "pro": 4096},
                                       # flash는 간단한 질문 전담이라 추론 사고를 꺼서 토큰 낭비를 줄이고,
                                       # pro로 승격되는 건 애초에 복잡한 질문이라 추론 예산을 그대로 둔다.
                                       thinking_budget={"flash": 0, "pro": None})
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
                                       MAIL_INSTRUCTION, MAIL_WELCOME,
                                       temperature=0.4, max_output_tokens=2048, thinking_budget=0)
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
                                       PUPPY_INSTRUCTION, PUPPY_WELCOME,
                                       max_output_tokens=1536, thinking_budget=0,
                                       # 그냥 가볍게 나누는 잡담용 봇이라, 다른 봇들과 달리 비용보다
                                       # "어지간한 건 다 기억하고 이어간다"를 우선한다.
                                       history_load_limit=60,
                                       session_max_turns=40,
                                       session_hard_limit_turns=60,
                                       session_token_soft_limit=50000,
                                       session_token_hard_limit=100000)
            t = threading.Thread(target=run_forever, args=(bot_obj, "개아"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"개아 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_CASUAL 없어서 개아는 건너뜁니다.")

    directory_token = _get_env_token("TELEGRAM_TOKEN_DIRECTORY")
    if directory_token:
        try:
            directory_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_DIRECTORY")), label="내선연락처")
            store = ChatHistoryStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="directory")
            kb_store = KnowledgeStore(UPSTASH_URL, UPSTASH_TOKEN, namespace="directory")
            bot_obj = MemoryGeminiBot("내선/비상연락 너구리", directory_token, directory_router, store,
                                       DIRECTORY_INSTRUCTION, DIRECTORY_WELCOME,
                                       knowledge_store=kb_store,
                                       temperature=0.2, max_output_tokens=1200, thinking_budget=0)
            t = threading.Thread(target=run_forever, args=(bot_obj, "내선/비상연락 너구리"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"내선/비상연락 너구리 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_DIRECTORY 없어서 내선/비상연락 너구리는 건너뜁니다.")

    docs_token = _get_env_token("TELEGRAM_TOKEN_DOCS")
    if docs_token:
        try:
            docs_router = GeminiRouter(genai.Client(api_key=resolve_api_key("GEMINI_API_KEY_DOCS")), label="서류")
            bot_obj = NeoguriDocumentBot("서류확인 너구리", docs_token, docs_router)
            t = threading.Thread(target=run_forever, args=(bot_obj, "서류확인 너구리"), daemon=True)
            t.start()
            threads.append(t)
        except Exception as e:
            logger.error(f"서류확인 너구리 초기화 실패, 건너뜁니다: {e}", exc_info=True)
    else:
        logger.warning("TELEGRAM_TOKEN_DOCS 없어서 서류확인 너구리는 건너뜁니다.")

    if not threads:
        raise ValueError("실행 가능한 봇이 없습니다. 환경변수를 확인하세요.")

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
