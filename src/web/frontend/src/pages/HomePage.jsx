import React, { useState, useEffect, useRef } from 'react'
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
  const { playlist, isConnected, systemStatus, previewData: wsPreviewData } = useWebSocket()
  const [ledPositions, setLedPositions] = useState(null)
  const canvasRef = useRef(null)

  const currentItem = playlist.items?.[playlist.current_index]
  const status = systemStatus

  // System status is now received via WebSocket, no need for HTTP polling

  // Fetch LED positions once on component mount
  useEffect(() => {
    const fetchLedPositions = async () => {
      try {
        const response = await fetch('/api/led-positions')
        if (response.ok) {
          const data = await response.json()
          setLedPositions(data)
          console.log('LED positions loaded:', data.led_count, 'LEDs')
          console.log('Debug stats:', data.debug)
          console.log('Frame dimensions:', data.frame_dimensions)
          console.log('First 3 positions:', data.positions.slice(0, 3))
        }
      } catch (error) {
        console.error('Failed to fetch LED positions:', error)
      }
    }

    fetchLedPositions()
  }, [])

  // Use WebSocket preview data (no need for HTTP polling)
  const previewData = wsPreviewData

  // Canvas-based LED rendering function
  const drawLEDs = () => {
    if (!canvasRef.current || !ledPositions) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    // Clear canvas with black background
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Set additive blending mode for LED rendering
    ctx.globalCompositeOperation = 'lighter'

    // Calculate scaling to fit LED coordinate space (800x480) into canvas
    const scaleX = canvas.width / ledPositions.frame_dimensions.width
    const scaleY = canvas.height / ledPositions.frame_dimensions.height

    console.log(`Canvas rendering: ${canvas.width}x${canvas.height}, scale: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`)

    // Draw each LED
    ledPositions.positions.forEach((position, i) => {
      const [x, y] = position

      // Scale coordinates to canvas size
      const canvasX = x * scaleX
      const canvasY = y * scaleY

      // Log first few LEDs for debugging
      if (i < 5) {
        console.log(`LED ${i}: raw pos [${x}, ${y}] -> canvas [${canvasX.toFixed(1)}, ${canvasY.toFixed(1)}]`)
      }

      // Determine LED color and skip dark LEDs
      if (previewData?.has_frame && previewData?.frame_data) {
        let colorData = previewData.frame_data[i]

        if (colorData && Array.isArray(colorData) && colorData.length >= 3) {
          const [r, g, b] = colorData

          // Skip completely dark LEDs to avoid unnecessary computation
          if (r === 0 && g === 0 && b === 0) {
            return
          }

          // Use direct RGB values for additive blending
          const ledColor = `rgb(${r}, ${g}, ${b})`

          // Debug log for first few LEDs
          if (i < 5 && (r > 0 || g > 0 || b > 0)) {
            console.log(`LED ${i} color: rgb(${r}, ${g}, ${b})`)
          }

          // Draw LED circle with additive blending
          ctx.beginPath()
          ctx.arc(canvasX, canvasY, 8, 0, 2 * Math.PI)
          ctx.fillStyle = ledColor
          ctx.fill()

          // Add glow effect for bright LEDs (any channel > 200)
          if (r > 200 || g > 200 || b > 200) {
            ctx.beginPath()
            ctx.arc(canvasX, canvasY, 12, 0, 2 * Math.PI)
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.3)`
            ctx.fill()
          }
        }
      } else {
        // Fallback: draw dim LEDs when no preview data
        ctx.beginPath()
        ctx.arc(canvasX, canvasY, 8, 0, 2 * Math.PI)
        ctx.fillStyle = 'rgba(102, 102, 102, 0.3)'
        ctx.fill()
      }
    })

    // Reset composite operation for any future drawing
    ctx.globalCompositeOperation = 'source-over'
  }

  // Redraw LEDs when data changes
  useEffect(() => {
    drawLEDs()
  }, [ledPositions, previewData])

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
        <div className="flex flex-wrap justify-center gap-3 mt-2">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-retro text-xs font-mono ${
            isConnected
              ? 'text-neon-green border border-neon-green border-opacity-30 bg-neon-green bg-opacity-5'
              : 'text-metal-silver border border-metal-silver border-opacity-30 bg-dark-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-neon-green animate-pulse-neon' : 'bg-metal-silver'
            }`} />
            {isConnected ? 'ONLINE' : 'OFFLINE'}
          </div>

          {/* LED Panel status */}
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-retro text-xs font-mono ${
            status?.led_panel_connected
              ? 'text-neon-cyan border border-neon-cyan border-opacity-30 bg-neon-cyan bg-opacity-5'
              : status?.led_panel_status === 'connecting'
              ? 'text-neon-orange border border-neon-orange border-opacity-30 bg-neon-orange bg-opacity-5'
              : 'text-metal-silver border border-metal-silver border-opacity-30 bg-dark-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              status?.led_panel_connected
                ? 'bg-neon-cyan animate-pulse-neon'
                : status?.led_panel_status === 'connecting'
                ? 'bg-neon-orange animate-pulse'
                : 'bg-metal-silver'
            }`} />
            {status?.led_panel_connected
              ? 'CONNECTED'
              : status?.led_panel_status === 'connecting'
              ? 'CONNECTING...'
              : 'DISCONNECTED'
            }
          </div>
        </div>
      </div>

      {/* LED Preview Area */}
      <div className="retro-container">
        <h2 className="text-lg font-retro text-neon-cyan mb-4 text-center">
          LED ARRAY PREVIEW
        </h2>

        <div className="relative aspect-[5/3] bg-dark-800 rounded-retro border border-neon-cyan border-opacity-20 overflow-hidden">
          {/* Canvas-based LED display */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
            width={800}
            height={480}
            style={{ imageRendering: 'pixelated' }}
          />

          {/* Loading indicator when LED positions not loaded */}
          {!ledPositions && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-neon-cyan border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                <p className="text-neon-cyan text-sm font-mono">Loading LED layout...</p>
              </div>
            </div>
          )}

          {/* Overlay text when not playing */}
          {!playlist.is_playing && ledPositions && (
            <div className="absolute inset-0 flex items-center justify-center bg-dark-800 bg-opacity-50">
              <div className="text-center">
                <SpeakerXMarkIcon className="w-12 h-12 text-metal-silver mx-auto mb-2 opacity-50" />
                <p className="text-metal-silver text-sm font-mono">
                  {currentItem ? 'PAUSED' : 'NO CONTENT'}
                </p>
                <p className="text-metal-silver text-xs font-mono mt-1 opacity-75">
                  {ledPositions.led_count} LEDs ready
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
              <span className="text-neon-cyan ml-2">{status.memory_usage?.toFixed(1)}% ({status.memory_usage_gb?.toFixed(1)}GB)</span>
            </div>
            <div>
              <span className="text-metal-silver">FPS:</span>
              <span className="text-neon-cyan ml-2">{status.frame_rate?.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-metal-silver">Late Frames:</span>
              <span className="text-neon-cyan ml-2">{status.late_frame_percentage?.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage
