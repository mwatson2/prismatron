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
  const [previewBrightness, setPreviewBrightness] = useState(0.5)
  const canvasRef = useRef(null)
  const ledStampRef = useRef(null)

  const currentItem = playlist.items?.[playlist.current_index]
  const status = systemStatus

  // Debug logging for playlist state updates
  useEffect(() => {
    console.log('Playlist state updated:', {
      itemCount: playlist.items?.length || 0,
      currentIndex: playlist.current_index,
      isPlaying: playlist.is_playing,
      currentItemName: currentItem?.name || 'none'
    })
  }, [playlist, currentItem])

  // System status is now received via WebSocket, no need for HTTP polling

  // Fetch LED positions and create stamps once on component mount
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

    // Create LED stamp for fast rendering
    createLEDStamp()
    fetchLedPositions()
  }, [])

  // Use WebSocket preview data (no need for HTTP polling)
  const previewData = wsPreviewData

  // Create pre-rendered LED stamp for fast rendering
  const createLEDStamp = () => {
    // LED stamp (24px radius = 48px diameter)
    const ledRadius = 24
    const ledSize = ledRadius * 2.5 * 2 // Extended for Gaussian falloff
    const ledCanvas = document.createElement('canvas')
    ledCanvas.width = ledCanvas.height = ledSize
    const ledCtx = ledCanvas.getContext('2d')

    // Create Gaussian falloff pattern
    const centerX = ledSize / 2
    const centerY = ledSize / 2

    // Create gradient that approximates Gaussian falloff
    const ledGradient = ledCtx.createRadialGradient(centerX, centerY, 0, centerX, centerY, ledRadius * 2.5)
    ledGradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)')     // Center: full intensity
    ledGradient.addColorStop(0.32, 'rgba(255, 255, 255, 0.8)')  // ~1 sigma: 80%
    ledGradient.addColorStop(0.6, 'rgba(255, 255, 255, 0.4)')   // ~2 sigma: 40%
    ledGradient.addColorStop(0.85, 'rgba(255, 255, 255, 0.1)')  // ~2.5 sigma: 10%
    ledGradient.addColorStop(1, 'rgba(255, 255, 255, 0)')       // Edge: transparent

    ledCtx.fillStyle = ledGradient
    ledCtx.fillRect(0, 0, ledSize, ledSize)
    ledStampRef.current = ledCanvas
  }

  // Fast canvas-based LED rendering using pre-rendered stamp
  const drawLEDs = () => {
    if (!canvasRef.current || !ledPositions || !ledStampRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const ledStamp = ledStampRef.current

    // Clear canvas with black background
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Calculate scaling to fit LED coordinate space (800x480) into canvas
    const scaleX = canvas.width / ledPositions.frame_dimensions.width
    const scaleY = canvas.height / ledPositions.frame_dimensions.height

    console.log(`Fast LED rendering: ${canvas.width}x${canvas.height}, scale: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`)

    // Draw each LED using pre-rendered stamp
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

          // Apply brightness factor to reduce saturation
          const adjustedR = Math.round(r * previewBrightness)
          const adjustedG = Math.round(g * previewBrightness)
          const adjustedB = Math.round(b * previewBrightness)

          // Debug log for first few LEDs
          if (i < 5 && (r > 0 || g > 0 || b > 0)) {
            console.log(`LED ${i} color: rgb(${r}, ${g}, ${b}) -> rgb(${adjustedR}, ${adjustedG}, ${adjustedB}) (brightness: ${previewBrightness})`)
          }

          // Draw LED using stamp technique
          ctx.save()
          ctx.globalCompositeOperation = 'screen' // Additive-like blending
          ctx.fillStyle = `rgb(${adjustedR}, ${adjustedG}, ${adjustedB})`
          ctx.fillRect(
            canvasX - ledStamp.width / 2,
            canvasY - ledStamp.height / 2,
            ledStamp.width,
            ledStamp.height
          )
          ctx.globalCompositeOperation = 'multiply'
          ctx.drawImage(
            ledStamp,
            canvasX - ledStamp.width / 2,
            canvasY - ledStamp.height / 2
          )
          ctx.restore()
        }
      } else {
        // Fallback: draw dim LEDs when no preview data using stamp
        ctx.save()
        ctx.globalCompositeOperation = 'screen'
        ctx.fillStyle = 'rgb(102, 102, 102)'
        ctx.fillRect(
          canvasX - ledStamp.width / 2,
          canvasY - ledStamp.height / 2,
          ledStamp.width,
          ledStamp.height
        )
        ctx.globalCompositeOperation = 'multiply'
        ctx.globalAlpha = 0.3
        ctx.drawImage(
          ledStamp,
          canvasX - ledStamp.width / 2,
          canvasY - ledStamp.height / 2
        )
        ctx.restore()
      }
    })
  }

  // Redraw LEDs when data changes
  useEffect(() => {
    drawLEDs()
  }, [ledPositions, previewData, previewBrightness])

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
                  {(currentItem || status?.current_file) ? 'PAUSED' : 'NO CONTENT'}
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
      {(currentItem || status?.current_file) && (
        <div className="retro-container">
          <h3 className="text-sm font-retro text-neon-pink mb-2">NOW PLAYING</h3>
          <div className="space-y-2">
            <p className="text-neon-cyan font-medium truncate">
              {currentItem?.name || status?.current_file || 'Unknown'}
            </p>
            <div className="flex justify-between text-xs text-metal-silver font-mono">
              <span className="uppercase">{currentItem?.type || 'content'}</span>
              <div className="flex items-center gap-2">
                {currentItem?.duration && (
                  <span>{Math.round(currentItem.duration)}s</span>
                )}
                <span className={`px-2 py-1 rounded text-xs ${
                  playlist.is_playing
                    ? 'bg-neon-green bg-opacity-20 text-neon-green'
                    : 'bg-metal-silver bg-opacity-20 text-metal-silver'
                }`}>
                  {playlist.is_playing ? 'PLAYING' : 'PAUSED'}
                </span>
              </div>
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
              <span className="text-metal-silver">Input FPS:</span>
              <span className="text-neon-cyan ml-2">{status.consumer_input_fps?.toFixed(1) || 'N/A'}</span>
            </div>
            <div>
              <span className="text-metal-silver">Output FPS:</span>
              <span className="text-neon-cyan ml-2">{status.renderer_output_fps?.toFixed(1) || 'N/A'}</span>
            </div>
            <div>
              <span className="text-metal-silver">Late Frames:</span>
              <span className="text-neon-cyan ml-2">{status.late_frame_percentage?.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-metal-silver">Dropped Frames:</span>
              <span className={`ml-2 ${status.dropped_frames_percentage > 5 ? 'text-neon-orange' : 'text-neon-cyan'}`}>
                {status.dropped_frames_percentage?.toFixed(1) || '0.0'}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Preview Brightness Control */}
      <div className="retro-container">
        <h3 className="text-sm font-retro text-neon-purple mb-3">PREVIEW BRIGHTNESS</h3>
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <span className="text-xs text-metal-silver font-mono w-8">0%</span>
            <div className="flex-1 relative">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={previewBrightness}
                onChange={(e) => setPreviewBrightness(parseFloat(e.target.value))}
                className="w-full h-2 bg-dark-700 rounded-retro appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right,
                    #333 0%,
                    #333 ${previewBrightness * 100}%,
                    #1a1a1a ${previewBrightness * 100}%,
                    #1a1a1a 100%)`
                }}
              />
              <div
                className="absolute top-0 h-2 bg-gradient-to-r from-neon-purple to-neon-cyan rounded-retro pointer-events-none"
                style={{ width: `${previewBrightness * 100}%` }}
              />
            </div>
            <span className="text-xs text-metal-silver font-mono w-12">100%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-metal-silver font-mono">
              Reduces LED saturation for clearer preview
            </span>
            <span className="text-sm text-neon-cyan font-mono">
              {Math.round(previewBrightness * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HomePage
