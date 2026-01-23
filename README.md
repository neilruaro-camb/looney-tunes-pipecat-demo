# CAMB AI Voice Demo

A simple voice agent demo powered by CAMB AI's TTS, Deepgram STT, and OpenAI GPT.

## Setup

### 1. Install dependencies

**Backend (using uv):**
```bash
cd backend
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
```

### 2. Configure environment

Copy `.env.example` to `backend/.env` and fill in your API keys:

```bash
cp .env.example backend/.env
```

Required API keys:
- `CAMB_API_KEY` - CAMB AI API key for TTS
- `DEEPGRAM_API_KEY` - Deepgram API key for STT
- `OPENAI_API_KEY` - OpenAI API key for GPT
- `DAILY_API_KEY` - Daily.co API key for WebRTC

### 3. Run the app

**Development (two terminals):**

Terminal 1 - Backend:
```bash
cd backend
uv run python server.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173

**Production:**

Build the frontend and serve from backend:
```bash
cd frontend
npm run build
cd ../backend
uv run python server.py
```

Then open http://localhost:7860
