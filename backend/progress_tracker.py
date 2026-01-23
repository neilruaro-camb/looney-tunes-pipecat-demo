"""Progress tracker processors for sending status updates to the frontend."""

import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
    OutputTransportMessageFrame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class STTProgressProcessor(FrameProcessor):
    """Tracks STT progress."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._user_message_id: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            await self._send_status("listening", frame.text)

        elif isinstance(frame, TranscriptionFrame):
            self._user_message_id += 1
            await self._send_status("stt", frame.text)
            await self._send_transcript("user", frame.text, final=True, message_id=self._user_message_id)
            await self._send_status("llm")

        await self.push_frame(frame, direction)

    async def _send_status(self, status: str, text: Optional[str] = None):
        message = {"type": "status", "status": status}
        if text:
            message["text"] = text
        await self.push_frame(
            OutputTransportMessageFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )

    async def _send_transcript(self, role: str, text: str, final: bool = True, message_id: Optional[int] = None):
        message = {
            "type": "transcript",
            "role": role,
            "text": text,
            "final": final,
            "timestamp": int(time.time() * 1000),
        }
        if message_id is not None:
            message["messageId"] = message_id
        await self.push_frame(
            OutputTransportMessageFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )


class LLMProgressProcessor(FrameProcessor):
    """Tracks LLM progress."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._assistant_text: str = ""
        self._assistant_message_id: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._assistant_text = ""
            self._assistant_message_id += 1

        elif isinstance(frame, LLMTextFrame):
            self._assistant_text += frame.text
            await self._send_transcript(
                "assistant",
                self._assistant_text,
                final=False,
                message_id=self._assistant_message_id
            )

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._assistant_text:
                await self._send_transcript(
                    "assistant",
                    self._assistant_text,
                    final=True,
                    message_id=self._assistant_message_id
                )
                self._assistant_text = ""

        await self.push_frame(frame, direction)

    async def _send_transcript(self, role: str, text: str, final: bool = True, message_id: Optional[int] = None):
        message = {
            "type": "transcript",
            "role": role,
            "text": text,
            "final": final,
            "timestamp": int(time.time() * 1000),
        }
        if message_id is not None:
            message["messageId"] = message_id
        await self.push_frame(
            OutputTransportMessageFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )


class TTSStatusProcessor(FrameProcessor):
    """Processor to track TTS status and interruptions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            if self._is_speaking:
                self._is_speaking = False
                await self.push_frame(
                    OutputTransportMessageFrame(message={"type": "status", "status": "idle"}),
                    FrameDirection.DOWNSTREAM,
                )

        elif isinstance(frame, TTSStartedFrame):
            if not self._is_speaking:
                self._is_speaking = True
                await self.push_frame(
                    OutputTransportMessageFrame(message={"type": "status", "status": "tts"}),
                    FrameDirection.DOWNSTREAM,
                )

        elif isinstance(frame, TTSStoppedFrame):
            if self._is_speaking:
                self._is_speaking = False
                await self.push_frame(
                    OutputTransportMessageFrame(message={"type": "status", "status": "idle"}),
                    FrameDirection.DOWNSTREAM,
                )

        elif isinstance(frame, TTSSpeakFrame):
            if not self._is_speaking:
                self._is_speaking = True
                await self.push_frame(
                    OutputTransportMessageFrame(message={"type": "status", "status": "tts"}),
                    FrameDirection.DOWNSTREAM,
                )

        await self.push_frame(frame, direction)
