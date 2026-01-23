"""Pipecat bot for CAMB AI voice demo."""

import os
from typing import Optional

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
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


def create_tts_service(model: str = "mars-flash") -> CambTTSService:
    """Create a TTS service with shared API client."""
    tts = CambTTSService(
        api_key=os.getenv("CAMB_API_KEY"),
        model=model,
        voice_id="156549"
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


SYSTEM_PROMPT = """You are a friendly and helpful voice assistant powered by CAMB AI.

Your personality:
- Warm, conversational, and engaging
- Helpful but concise
- Natural and human-like in your responses

Guidelines:
- Keep responses under 100 words since they will be spoken aloud
- Be conversational and friendly
- Answer questions helpfully on any topic
- If you don't know something, say so honestly

CRITICAL - Your responses will be read aloud by text-to-speech. You MUST:
- Never use asterisks (*), markdown formatting, or bullet points
- Never use special characters like #, -, _, or similar
- Never use parenthetical asides like (pause) or (laughs)
- Write in plain, flowing sentences only
- Spell out abbreviations and acronyms when first used
- Use words like "first", "second", "third" instead of numbered lists
"""


async def run_bot(room_url: str, token: str, tts_model: str = "mars-flash"):
    """Run the voice agent bot."""
    transport = DailyTransport(
        room_url,
        token,
        "CAMB AI Assistant",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)),
        ),
    )

    stt = create_stt_service()
    tts = create_tts_service(tts_model)
    llm = create_llm_service()

    logger.info(f"Using TTS model: {tts_model}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Greet me warmly and let me know you're here to help with anything I'd like to talk about.",
        },
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
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant['id']}, reason: {reason}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info("Bot finished")
