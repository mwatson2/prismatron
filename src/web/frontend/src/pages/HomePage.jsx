import React, { useState, useEffect } from 'react'
import {
  PlayIcon,
  PauseIcon,
  ForwardIcon,
  BackwardIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon
} from '@heroicons/react/24/solid'
import { useWebSocket } from '../hooks/useWebSocket'

const HomePage = () => {
  const { playlist, isConnected, systemStatus } = useWebSocket()
  const [localStatus, setLocalStatus] = useState(null)

  useEffect(() => {
    // Fetch initial system status
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/status')
        if (response.ok) {
          const status = await response.json()
          setLocalStatus(status)
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [])

  const currentItem = playlist.items?.[playlist.current_index]
  const status = systemStatus || localStatus

  const handlePlayPause = async () => {
    try {
      const endpoint = playlist.is_playing ? '/api/control/pause' : '/api/control/play'
      await fetch(endpoint, { method: 'POST' })
    } catch (error) {
      console.error('Failed to toggle playback:', error)
    }
  }

  const handleNext = async () => {
    try {
      await fetch('/api/control/next', { method: 'POST' })
    } catch (error) {
      console.error('Failed to skip to next:', error)
    }
  }

  const handlePrevious = async () => {
    try {
      await fetch('/api/control/previous', { method: 'POST' })
    } catch (error) {
      console.error('Failed to skip to previous:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-3xl font-retro font-bold text-neon text-neon-strong prismatron-logo">
          PRISMATRON
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          LED DISPLAY CONTROL INTERFACE
        </p>

        {/* Connection status */}
        <div className={`inline-flex items-center gap-2 mt-2 px-3 py-1 rounded-retro text-xs font-mono ${
          isConnected
            ? 'text-neon-green border border-neon-green border-opacity-30 bg-neon-green bg-opacity-5'
            : 'text-metal-silver border border-metal-silver border-opacity-30 bg-dark-800'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-neon-green animate-pulse-neon' : 'bg-metal-silver'
          }`} />
          {isConnected ? 'ONLINE' : 'OFFLINE'}
        </div>
      </div>

      {/* LED Preview Area */}
      <div className="retro-container">
        <h2 className="text-lg font-retro text-neon-cyan mb-4 text-center">
          LED ARRAY PREVIEW
        </h2>

        <div className="relative aspect-[5/3] bg-dark-800 rounded-retro border border-neon-cyan border-opacity-20 overflow-hidden">
          {/* Simulated LED grid */}
          <div className="absolute inset-0 grid grid-cols-20 gap-1 p-2">
            {Array.from({ length: 60 }, (_, i) => (
              <div
                key={i}
                className={`aspect-square rounded-full transition-all duration-200 ${
                  playlist.is_playing
                    ? 'bg-neon-cyan animate-pulse-neon shadow-neon'
                    : 'bg-metal-silver opacity-30'
                }`}
                style={{
                  animationDelay: `${(i * 50) % 2000}ms`
                }}
              />
            ))}
          </div>

          {/* Overlay text when not playing */}
          {!playlist.is_playing && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <SpeakerXMarkIcon className="w-12 h-12 text-metal-silver mx-auto mb-2 opacity-50" />
                <p className="text-metal-silver text-sm font-mono">
                  {currentItem ? 'PAUSED' : 'NO CONTENT'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Now Playing Info */}
      {currentItem && (
        <div className="retro-container">
          <h3 className="text-sm font-retro text-neon-pink mb-2">NOW PLAYING</h3>
          <div className="space-y-2">
            <p className="text-neon-cyan font-medium truncate">{currentItem.name}</p>
            <div className="flex justify-between text-xs text-metal-silver font-mono">
              <span className="uppercase">{currentItem.type}</span>
              {currentItem.duration && (
                <span>{Math.round(currentItem.duration)}s</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Playback Controls */}
      <div className="retro-container">
        <div className="flex items-center justify-center space-x-6">
          <button
            onClick={handlePrevious}
            disabled={!playlist.items?.length}
            className="retro-button p-3 rounded-full text-neon-cyan disabled:text-metal-silver disabled:cursor-not-allowed"
            aria-label="Previous track"
          >
            <BackwardIcon className="w-6 h-6" />
          </button>

          <button
            onClick={handlePlayPause}
            disabled={!playlist.items?.length}
            className="retro-button p-4 rounded-full text-neon-cyan text-neon disabled:text-metal-silver disabled:cursor-not-allowed"
            aria-label={playlist.is_playing ? 'Pause' : 'Play'}
          >
            {playlist.is_playing ? (
              <PauseIcon className="w-8 h-8" />
            ) : (
              <PlayIcon className="w-8 h-8" />
            )}
          </button>

          <button
            onClick={handleNext}
            disabled={!playlist.items?.length}
            className="retro-button p-3 rounded-full text-neon-cyan disabled:text-metal-silver disabled:cursor-not-allowed"
            aria-label="Next track"
          >
            <ForwardIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Playlist position indicator */}
        {playlist.items?.length > 0 && (
          <div className="mt-4 text-center">
            <p className="text-xs text-metal-silver font-mono">
              {playlist.current_index + 1} / {playlist.items.length}
            </p>

            {/* Progress bar */}
            <div className="mt-2 w-full bg-dark-700 rounded-retro h-1 overflow-hidden">
              <div
                className="h-full bg-neon-cyan transition-all duration-300"
                style={{
                  width: `${((playlist.current_index + 1) / playlist.items.length) * 100}%`
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* System Status */}
      {status && (
        <div className="retro-container">
          <h3 className="text-sm font-retro text-neon-orange mb-3">SYSTEM STATUS</h3>
          <div className="grid grid-cols-2 gap-4 text-xs font-mono">
            <div>
              <span className="text-metal-silver">CPU:</span>
              <span className="text-neon-cyan ml-2">{status.cpu_usage?.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-metal-silver">Memory:</span>
              <span className="text-neon-cyan ml-2">{status.memory_usage?.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-metal-silver">FPS:</span>
              <span className="text-neon-cyan ml-2">{status.frame_rate?.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-metal-silver">Uptime:</span>
              <span className="text-neon-cyan ml-2">
                {Math.floor((status.uptime || 0) / 3600)}h
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage
