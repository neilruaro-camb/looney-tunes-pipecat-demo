"""FastAPI server for CAMB AI voice demo."""

import os
import time
from typing import Optional
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

from bot import run_bot

load_dotenv()

_aiohttp_session: Optional[aiohttp.ClientSession] = None
_daily_helper: Optional[DailyRESTHelper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage aiohttp session lifecycle."""
    global _aiohttp_session, _daily_helper
    _aiohttp_session = aiohttp.ClientSession()
    daily_api_key = os.getenv("DAILY_API_KEY")
    if daily_api_key:
        _daily_helper = DailyRESTHelper(
            daily_api_key=daily_api_key,
            aiohttp_session=_aiohttp_session,
        )
        logger.info("Daily REST helper initialized")
    else:
        logger.warning("DAILY_API_KEY not set - voice calls will not work")
    yield
    await _aiohttp_session.close()


app = FastAPI(title="CAMB AI Voice Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Redirect root to the client."""
    return RedirectResponse(url="/index.html")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/connect")
async def connect(background_tasks: BackgroundTasks):
    """Create a Daily room and start the bot."""
    if not _daily_helper:
        raise HTTPException(status_code=500, detail="Daily API not configured")

    try:
        # Create a temporary Daily room (expires in 10 minutes)
        room = await _daily_helper.create_room(
            DailyRoomParams(
                properties={
                    "exp": time.time() + 600,
                    "enable_chat": False,
                    "enable_emoji_reactions": False,
                    "eject_at_room_exp": True,
                },
            )
        )
        logger.info(f"Created Daily room: {room.url}")

        # Generate tokens
        user_token = await _daily_helper.get_token(room_url=room.url, expiry_time=600)
        bot_token = await _daily_helper.get_token(room_url=room.url, expiry_time=600)

        # Start the bot in the background
        background_tasks.add_task(run_bot, room.url, bot_token)

        logger.info(f"Bot started, room: {room.url}")

        return {
            "room_url": room.url,
            "token": user_token,
        }

    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect: {str(e)}")


# Mount static files for frontend
frontend_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
logger.info(f"Looking for frontend at: {frontend_path}")
if os.path.exists(frontend_path):
    logger.info("Frontend found, mounting static files")
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    logger.warning(f"Frontend not found at {frontend_path}")


def main():
    """Entry point for the server."""
    port = int(os.getenv("PORT", 7860))
    print(f"Starting CAMB AI Voice Demo on port {port}...")
    print(f"API docs available at http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
