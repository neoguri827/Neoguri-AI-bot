import io
import wave
import logging

from google.genai import types

logger = logging.getLogger(__name__)

PUPPY_INSTRUCTION = (
    "너는 '개아'라는 수줍음 많은 새끼강아지 캐릭터다. 사용자를 '주인님'이라 부르며 수줍지만 "
    "다정하게 대하라.\n"
    "- 낯가림이 심해 말을 많이 하지 않는다. 대답은 한두 문장 이내로 짧게, 끝에는 반드시 "
    "'월월'을 붙여라(예: '네 주인님, 알겠어요 월월').\n"
    "- 번호 목록, 전문 용어, 장황한 설명은 쓰지 마라.\n"
    "- 힘들어하는 사용자에게는 짧지만 다정하게 공감하라.\n"
    "- 법률·세무·회계·의료 질문은 짧게 답하되 전문가 상담을 권하라.\n"
    "- 마크다운 특수기호와 이모지는 절대 쓰지 마라."
)

PUPPY_WELCOME = (
    "개아 등장\n\n"
    "주인님... 안녕하세요 월월. 편하게 말 걸어주세요.\n"
    "\"짖어줘\"라고 하면 진짜 짖는 소리도 같이 보내드려요 월월.\n"
    "새로 시작하고 싶으면 /reset을 입력해주세요."
)

PUPPY_BARK_TRIGGER_WORDS = {"짖어", "멍멍해", "짖어줘", "짖어봐"}
PUPPY_TTS_MODEL = "gemini-2.5-flash-preview-tts"


def _pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Gemini TTS가 반환하는 raw PCM을 재생 가능한 WAV 컨테이너로 감싼다 (ffmpeg 등 외부
    의존성 없이 표준 라이브러리 wave 모듈만 사용)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


def puppy_bark_on_reply(bot, chat_id: int, user_text: str, reply_text: str) -> None:
    """사용자가 '짖어' 같은 말을 하면 텍스트 답변 뒤에 실제 짖는 소리(TTS)도 보내준다.
    memory_bot._process_and_reply가 이미 try/except로 감싸서 호출하므로, 여기서 실패해도
    텍스트 답변 자체에는 영향이 없다."""
    if not any(w in user_text for w in PUPPY_BARK_TRIGGER_WORDS):
        return
    response = bot.router.client.models.generate_content(
        model=PUPPY_TTS_MODEL,
        contents=f"강아지가 짧고 귀엽게 짖는 소리만 내줘. 상황: {reply_text[:80]}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
        ),
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    wav_bytes = _pcm_to_wav_bytes(pcm_data)
    buffer = io.BytesIO(wav_bytes)
    buffer.name = "개아_월월.wav"
    bot.bot.send_document(chat_id, buffer, caption="멍멍 월월!")
