import { useState, useCallback, useRef, useEffect } from 'react'
import Daily, { DailyCall } from '@daily-co/daily-js'
import { Mic, MicOff, Phone, PhoneOff } from 'lucide-react'

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error'
type AgentState = 'listening' | 'thinking' | 'speaking' | 'idle'

type Character = {
  id: string
  name: string
  color: string
  avatar: string
}

type Theme = 'dark' | 'white' | 'transparent'

function getTheme(): Theme {
  const path = window.location.pathname
  if (path.startsWith('/white')) return 'white'
  if (path.startsWith('/transparent')) return 'transparent'
  return 'dark'
}

const THEMES = {
  dark: {
    bg: '#0d0d0d',
    card: '#171717',
    border: '#262626',
    text: '#ffffff',
    textMuted: '#9ca3af',
    textDimmer: '#6b7280',
  },
  white: {
    bg: '#ffffff',
    card: '#f5f5f5',
    border: '#e5e5e5',
    text: '#171717',
    textMuted: '#6b7280',
    textDimmer: '#9ca3af',
  },
  transparent: {
    bg: 'transparent',
    card: 'transparent',
    border: 'transparent',
    text: '#ffffff',
    textMuted: '#9ca3af',
    textDimmer: '#6b7280',
  },
}

const theme = getTheme()
const t = THEMES[theme]

// Apply theme as CSS custom properties before first render
const _root = document.documentElement
_root.style.setProperty('--t-bg', t.bg)
_root.style.setProperty('--t-card', t.card)
_root.style.setProperty('--t-border', t.border)
_root.style.setProperty('--t-text', t.text)
_root.style.setProperty('--t-text-muted', t.textMuted)
_root.style.setProperty('--t-text-dimmer', t.textDimmer)
document.body.style.backgroundColor = t.bg
document.body.style.color = t.text

const CHARACTERS: Character[] = [
  { id: 'bugs', name: 'Bugs Bunny', color: '#E8784A', avatar: 'https://cambai-not-ai.conbersa.ai/bugs-bunny.png' },
  { id: 'lola', name: 'Lola Bunny', color: '#A0A0A0', avatar: 'https://cambai-not-ai.conbersa.ai/lola-bunny.png' },
  { id: 'daffy', name: 'Daffy Duck', color: '#808080', avatar: 'https://cambai-not-ai.conbersa.ai/daffy-duck.png' },
]

function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle')
  const [agentState, setAgentState] = useState<AgentState>('idle')
  const [isMuted, setIsMuted] = useState(false)
  const [transcript, setTranscript] = useState<string>('')
  const [transcriptRole, setTranscriptRole] = useState<string>('')
  const [selectedCharacter, setSelectedCharacter] = useState<string>('bugs')
  const callRef = useRef<DailyCall | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const connect = useCallback(async () => {
    // Prevent double-clicks
    if (connectionState !== 'idle' && connectionState !== 'error') return

    setConnectionState('connecting')

    try {
      const response = await fetch('/api/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character: selectedCharacter }),
      })

      if (!response.ok) {
        throw new Error('Failed to create room')
      }

      const { room_url, token } = await response.json()
      console.log('[Daily] Got room URL:', room_url)

      // Create Daily call object
      const callObject = Daily.createCallObject({
        audioSource: true,
        videoSource: false,
      })
      callRef.current = callObject

      // Set up event handlers
      callObject.on('joined-meeting', () => {
        console.log('[Daily] Joined meeting')
        setConnectionState('connected')
        setAgentState('idle')
      })

      callObject.on('left-meeting', () => {
        console.log('[Daily] Left meeting')
        setConnectionState('idle')
        setAgentState('idle')
        setTranscript('')
        setTranscriptRole('')
      })

      callObject.on('error', (error) => {
        console.error('[Daily] Error:', error)
        setConnectionState('error')
      })

      // Handle remote audio
      callObject.on('track-started', (event) => {
        if (event.track?.kind === 'audio' && event.participant && !event.participant.local) {
          console.log('[Daily] Remote audio track started')
          // Clean up previous audio element if any
          if (audioRef.current) {
            audioRef.current.pause()
            audioRef.current.srcObject = null
          }
          const audio = new Audio()
          audio.srcObject = new MediaStream([event.track])
          audio.autoplay = true
          audioRef.current = audio
          audio.play().catch(console.error)
        }
      })

      // Handle app messages from the bot
      callObject.on('app-message', (event) => {
        if (event.data) {
          const data = event.data as { type: string; status?: string; text?: string; role?: string }
          console.log('[Daily] App message:', data)

          if (data.type === 'status') {
            switch (data.status) {
              case 'listening':
              case 'stt':
                setAgentState('listening')
                break
              case 'llm':
                setAgentState('thinking')
                break
              case 'tts':
                setAgentState('speaking')
                break
              default:
                setAgentState('idle')
            }
          }

          if (data.type === 'transcript' && data.text) {
            setTranscript(data.text)
            setTranscriptRole(data.role || 'assistant')
          }
        }
      })

      // Join the room
      await callObject.join({ url: room_url, token })

    } catch (error) {
      console.error('Connection error:', error)
      setConnectionState('error')
    }
  }, [connectionState, selectedCharacter])

  const disconnect = useCallback(async () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.srcObject = null
      audioRef.current = null
    }
    if (callRef.current) {
      await callRef.current.leave()
      await callRef.current.destroy()
      callRef.current = null
    }
    setConnectionState('idle')
    setAgentState('idle')
    setTranscript('')
    setTranscriptRole('')
  }, [])

  const toggleMute = useCallback(() => {
    if (callRef.current) {
      const currentMuteState = callRef.current.localAudio() === false
      callRef.current.setLocalAudio(currentMuteState)
      setIsMuted(!currentMuteState)
      console.log('[Daily] Microphone muted:', !currentMuteState)
    }
  }, [])

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.srcObject = null
      }
      if (callRef.current) {
        callRef.current.leave()
        callRef.current.destroy()
      }
    }
  }, [])

  return (
    <div className="min-h-screen bg-[var(--t-bg)] flex flex-col items-center justify-center p-4">
      {/* Character Selection — pill tabs */}
      <div className="flex flex-wrap gap-2 mb-6 justify-center">
        {CHARACTERS.map((char) => {
          const isSelected = selectedCharacter === char.id
          const isDisabled = connectionState === 'connecting' || connectionState === 'connected'
          return (
            <button
              key={char.id}
              onClick={() => !isDisabled && setSelectedCharacter(char.id)}
              disabled={isDisabled}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200
                ${isDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
                ${isSelected
                  ? 'bg-camb-orange/15 text-camb-orange'
                  : 'bg-[var(--t-card)] text-[var(--t-text-muted)] hover:text-[var(--t-text)]'
                }
              `}
            >
              <img
                src={char.avatar}
                alt={char.name}
                className="w-5 h-6 object-contain flex-shrink-0"
              />
              {char.name}
            </button>
          )
        })}
      </div>

      {/* Transcript / Status Display */}
      <div className="max-w-2xl w-full rounded-2xl bg-[var(--t-card)] border border-[var(--t-border)] min-h-[160px] flex items-center justify-center px-6 py-8 mb-6">
        {connectionState === 'idle' && (
          <p className="text-[var(--t-text-dimmer)] text-center">Tap to start the call...</p>
        )}
        {connectionState === 'connecting' && (
          <p className="text-[var(--t-text-muted)] text-center">Connecting...</p>
        )}
        {connectionState === 'error' && (
          <p className="text-red-400 text-center">Connection failed. Tap to retry.</p>
        )}
        {connectionState === 'connected' && (
          <>
            {!transcript && (agentState === 'listening' || agentState === 'idle') && (
              <p className="text-[var(--t-text-muted)] text-center">Listening...</p>
            )}
            {!transcript && agentState === 'thinking' && (
              <p className="text-[var(--t-text-muted)] text-center">Thinking...</p>
            )}
            {transcript && (
              <p className={`text-center text-lg ${transcriptRole === 'user' ? 'text-camb-orange' : 'text-[var(--t-text)]'}`}>
                {transcript}
              </p>
            )}
          </>
        )}
      </div>

      {/* Call Controls */}
      <div className="flex items-center gap-4">
        {(connectionState === 'idle' || connectionState === 'error') && (
          <button
            onClick={connect}
            className="w-14 h-14 rounded-full bg-camb-orange hover:bg-camb-orange/90 flex items-center justify-center transition-all duration-200 hover:scale-105 shadow-lg shadow-black/30"
          >
            <Phone className="w-6 h-6 text-white" />
          </button>
        )}
        {connectionState === 'connecting' && (
          <>
            <button
              disabled
              className="w-12 h-12 rounded-full bg-[var(--t-card)] flex items-center justify-center cursor-wait"
            >
              <Mic className="w-5 h-5 text-[var(--t-text-dimmer)]" />
            </button>
            <button
              onClick={disconnect}
              className="w-12 h-12 rounded-full bg-red-500 hover:bg-red-600 flex items-center justify-center transition-all duration-200"
            >
              <PhoneOff className="w-5 h-5 text-white" />
            </button>
          </>
        )}
        {connectionState === 'connected' && (
          <>
            <button
              onClick={toggleMute}
              className={`
                w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200
                ${isMuted
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                  : agentState === 'listening'
                    ? 'bg-white text-black hover:bg-gray-200'
                    : 'bg-[var(--t-card)] text-[var(--t-text-muted)] hover:bg-[var(--t-border)] hover:text-[var(--t-text)]'
                }
              `}
            >
              {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>
            <button
              onClick={disconnect}
              className="w-12 h-12 rounded-full bg-red-500 hover:bg-red-600 flex items-center justify-center transition-all duration-200"
            >
              <PhoneOff className="w-5 h-5 text-white" />
            </button>
          </>
        )}
      </div>

    </div>
  )
}

export default App
