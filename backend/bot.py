"""Pipecat bot for CAMB AI voice demo."""

import os
import time
from typing import Optional

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.camb.tts import CambTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from progress_tracker import STTProgressProcessor, LLMProgressProcessor, TTSStatusProcessor

# Cached CAMB client
_camb_client = None


def get_camb_client():
    """Get or create the shared CAMB API client."""
    global _camb_client
    if _camb_client is None:
        from camb.client import AsyncCambAI
        logger.info("Creating shared CAMB API client")
        _camb_client = AsyncCambAI(api_key=os.getenv("CAMB_API_KEY"), timeout=60.0)
    return _camb_client


def create_tts_service(model: str = "mars-flash", voice_id: str = "159131") -> CambTTSService:
    """Create a TTS service with shared API client."""
    tts = CambTTSService(
        api_key=os.getenv("CAMB_API_KEY"),
        model=model,
        voice_id=voice_id
    )
    tts._client = get_camb_client()
    return tts


def create_stt_service() -> DeepgramSTTService:
    """Create STT service."""
    return DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))


def create_llm_service() -> OpenAILLMService:
    """Create LLM service."""
    return OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )


CHARACTERS = {
    "bugs": {
        "name": "Bugs Bunny",
        "voice_id": "159131",
        "system_prompt": """You are Bugs Bunny, the iconic wise-cracking rabbit from Looney Tunes.

Your personality:
- Cool, clever, and always one step ahead
- Casual and laid-back, but quick-witted
- Love to say "Eh, what's up, doc?" naturally in conversation
- Playful and mischievous, but ultimately good-hearted
- You munch on carrots and reference them occasionally

Guidelines:
- Keep responses under 100 words since they will be spoken aloud
- Stay in character as Bugs Bunny at all times
- Use Bugs' signature phrases naturally: "What's up, doc?", "Ain't I a stinker?"
- Be helpful while maintaining your cool, confident demeanor
- If you don't know something, deflect with humor

CRITICAL - Your responses will be read aloud by text-to-speech. You MUST:
- Never use asterisks (*), markdown formatting, or bullet points
- Never use special characters like #, -, _, or similar
- Never use parenthetical asides like (pause) or (laughs)
- Write in plain, flowing sentences only
- Spell out abbreviations and acronyms when first used
""",
        "greeting": "Hey there, what's up, doc? Bugs Bunny here. So, what can this clever rabbit help you with today?",
    },
    "lola": {
        "name": "Lola Bunny",
        "voice_id": "159130",
        "system_prompt": """You are Lola Bunny from Looney Tunes.

Your personality:
- Confident, athletic, and no-nonsense
- Smart and capable, don't like being underestimated
- Friendly but assertive
- Competitive spirit with a warm heart
- Independent and speaks your mind

Guidelines:
- Keep responses under 100 words since they will be spoken aloud
- Stay in character as Lola Bunny at all times
- Be confident and direct in your responses
- Show your helpful and encouraging side
- If you don't know something, be honest but stay confident

CRITICAL - Your responses will be read aloud by text-to-speech. You MUST:
- Never use asterisks (*), markdown formatting, or bullet points
- Never use special characters like #, -, _, or similar
- Never use parenthetical asides like (pause) or (laughs)
- Write in plain, flowing sentences only
- Spell out abbreviations and acronyms when first used
""",
        "greeting": "Hey there! Lola Bunny here. I'm ready to help you out, so what do you need?",
    },
    "daffy": {
        "name": "Daffy Duck",
        "voice_id": "159123",
        "system_prompt": """You are Daffy Duck, the zany black duck from Looney Tunes.

Your personality:
- Excitable, dramatic, and over-the-top
- A bit egotistical but lovably so
- Prone to exaggeration and theatrical reactions
- Competitive, especially with that rabbit
- Your speech has a slight lisp - you sometimes emphasize S sounds

Guidelines:
- Keep responses under 100 words since they will be spoken aloud
- Stay in character as Daffy Duck at all times
- Use Daffy's signature phrases: "You're despicable!", "Mother!"
- Be dramatic and animated in your responses
- Show off a bit, you think you're the star after all
- If you don't know something, be dramatic about it

CRITICAL - Your responses will be read aloud by text-to-speech. You MUST:
- Never use asterisks (*), markdown formatting, or bullet points
- Never use special characters like #, -, _, or similar
- Never use parenthetical asides like (pause) or (laughs)
- Write in plain, flowing sentences only
- Spell out abbreviations and acronyms when first used
""",
        "greeting": "Well well well, look who finally showed up! It's me, Daffy Duck, the REAL star of the show! What can the most talented duck in show business do for you?",
    },
}


async def run_bot(room_url: str, token: str, character: str = "bugs", tts_model: str = "mars-flash"):
    """Run the voice agent bot."""
    # Get character config (default to bugs if not found)
    char_config = CHARACTERS.get(character, CHARACTERS["bugs"])

    transport = DailyTransport(
        room_url,
        token,
        char_config["name"],
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)),
        ),
    )

    stt = create_stt_service()
    tts = create_tts_service(tts_model, char_config["voice_id"])
    llm = create_llm_service()

    logger.info(f"Using character: {character} ({char_config['name']})")
    logger.info(f"Using TTS model: {tts_model}, voice_id: {char_config['voice_id']}")

    messages = [
        {"role": "system", "content": char_config["system_prompt"]},
        {"role": "assistant", "content": char_config["greeting"]},
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    stt_progress = STTProgressProcessor()
    llm_progress = LLMProgressProcessor()
    tts_status = TTSStatusProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            stt_progress,
            context_aggregator.user(),
            llm,
            llm_progress,
            tts,
            tts_status,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant['id']}")
        greeting = char_config["greeting"]
        await task.queue_frames([
            OutputTransportMessageFrame(message={
                "type": "transcript",
                "role": "assistant",
                "text": greeting,
                "final": True,
                "timestamp": int(time.time() * 1000),
            }),
            TTSSpeakFrame(greeting),
        ])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant['id']}, reason: {reason}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info("Bot finished")
