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

  // Format duration helper function
  const formatDuration = (seconds) => {
    if (!seconds || seconds < 0) return '--:--'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }
  const [ledPositions, setLedPositions] = useState(null)
  const [previewBrightness, setPreviewBrightness] = useState(0.8)
  const [optimizationIterations, setOptimizationIterations] = useState(null)
  const [previewFps, setPreviewFps] = useState(0)
  const lastRenderTimeRef = useRef(null)
  const ewmaIntervalRef = useRef(null)
  const renderStatsRef = useRef({ ewmaRenderDuration: null, renderCount: 0, lastLogTime: null })
  const canvasRef = useRef(null)
  const ledStampRRef = useRef(null)
  const ledStampGRef = useRef(null)
  const ledStampBRef = useRef(null)

  // Use WebSocket preview data (no need for HTTP polling)
  const previewData = wsPreviewData
  const status = systemStatus

  // Use ONLY preview data's current_item (based on rendering_index) for HomePage display
  // No fallbacks - if this is missing, we have a bug that needs to be fixed
  const currentItem = previewData?.current_item ? {
    name: previewData.current_item.name,
    type: previewData.current_item.type
  } : null

  // Track when the current rendered item actually changes
  useEffect(() => {
    const itemName = previewData?.current_item?.name || null;
    if (itemName) {
      console.log(`📺 RENDERED ITEM CHANGED: "${itemName}" (rendering_index: ${status?.rendering_index ?? 'undefined'})`);
    } else if (playlist.is_playing) {
      console.warn('🚨 PLAYING BUT NO CURRENT ITEM in previewData');
    }
  }, [previewData?.current_item?.name])

  // Removed spammy playback position logging

  // Track rendering index changes separately
  useEffect(() => {
    if (status?.rendering_index !== undefined) {
      console.log(`🎯 RENDERING INDEX CHANGED: ${status.rendering_index}`);
    }
  }, [status?.rendering_index])

  // Initialize optimization iterations from system status
  useEffect(() => {
    if (status?.optimization_iterations !== undefined && optimizationIterations === null) {
      setOptimizationIterations(status.optimization_iterations);
      console.log(`Initialized optimization iterations slider: ${status.optimization_iterations}`);
    }
  }, [status?.optimization_iterations, optimizationIterations])

  // System status is now received via WebSocket, no need for HTTP polling

  // Fetch LED positions and create stamps once on component mount
  useEffect(() => {
    const fetchLedPositions = async () => {
      try {
        const response = await fetch('/api/led-positions')
        if (response.ok) {
          const data = await response.json()
          setLedPositions(data)
          console.debug('LED positions loaded:', data.led_count, 'LEDs')
        }
      } catch (error) {
        console.error('Failed to fetch LED positions:', error)
      }
    }

    // Create LED stamps for fast rendering
    createLEDStamps()
    fetchLedPositions()
  }, [])

  // Create pre-rendered LED stamps for fast rendering with proper Gaussian shape
  const createLEDStamps = () => {
    // LED stamp dimensions - restored to original size for better visual quality
    const ledRadius = 9
    const ledSize = ledRadius * 2.5 * 2 // Restored to 2.5 for better appearance
    const centerX = ledSize / 2
    const centerY = ledSize / 2

    // Helper function to create a stamp for a specific color channel
    const createColorStamp = (red, green, blue) => {
      const canvas = document.createElement('canvas')
      canvas.width = canvas.height = ledSize
      const ctx = canvas.getContext('2d')

      // Create radial gradient with Gaussian-like falloff
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, ledRadius * 2.5)
      gradient.addColorStop(0, `rgba(${red}, ${green}, ${blue}, 1.0)`)     // Center: full intensity
      gradient.addColorStop(0.32, `rgba(${red}, ${green}, ${blue}, 0.8)`)  // ~1 sigma: 80%
      gradient.addColorStop(0.6, `rgba(${red}, ${green}, ${blue}, 0.4)`)   // ~2 sigma: 40%
      gradient.addColorStop(0.85, `rgba(${red}, ${green}, ${blue}, 0.1)`)  // ~2.5 sigma: 10%
      gradient.addColorStop(1, `rgba(${red}, ${green}, ${blue}, 0)`)       // Edge: transparent

      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, ledSize, ledSize)
      return canvas
    }

    // Create separate stamps for R, G, B channels
    ledStampRRef.current = createColorStamp(255, 0, 0)  // Pure red
    ledStampGRef.current = createColorStamp(0, 255, 0)  // Pure green
    ledStampBRef.current = createColorStamp(0, 0, 255)  // Pure blue
  }

  // Convert linear light value to sRGB for display
  // LED values from backend are in linear space, but canvas expects sRGB
  const linearToSrgb = (linear) => {
    // Piecewise sRGB transfer function
    // For values <= 0.0031308: sRGB = linear * 12.92
    // For values > 0.0031308: sRGB = 1.055 * linear^(1/2.4) - 0.055
    return linear <= 0.0031308
      ? linear * 12.92
      : 1.055 * Math.pow(linear, 1.0 / 2.4) - 0.055
  }

  // Fast canvas-based LED rendering using pre-rendered RGB stamps
  const drawLEDs = () => {
    if (!canvasRef.current || !ledPositions || !ledStampRRef.current || !ledStampGRef.current || !ledStampBRef.current) return

    const renderStartTime = performance.now()

    // Calculate preview FPS using EWMA of inter-render intervals
    if (lastRenderTimeRef.current !== null) {
      const interval = renderStartTime - lastRenderTimeRef.current

      // Use EWMA with alpha=0.1 for smooth FPS display
      if (ewmaIntervalRef.current === null) {
        ewmaIntervalRef.current = interval
      } else {
        ewmaIntervalRef.current = 0.1 * interval + 0.9 * ewmaIntervalRef.current
      }

      // Calculate FPS as reciprocal of interval (convert ms to seconds)
      const fps = 1000 / ewmaIntervalRef.current
      setPreviewFps(fps)
    }
    lastRenderTimeRef.current = renderStartTime

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const stampR = ledStampRRef.current
    const stampG = ledStampGRef.current
    const stampB = ledStampBRef.current

    // Reset composite operation before clearing to ensure proper clear
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1.0

    // Clear canvas with black background
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Calculate scaling to fit LED coordinate space (800x480) into canvas
    const scaleX = canvas.width / ledPositions.frame_dimensions.width
    const scaleY = canvas.height / ledPositions.frame_dimensions.height

    console.debug(`RGB Stamp LED rendering: ${canvas.width}x${canvas.height}, scale: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`)

    // Use additive blending to properly composite RGB stamps
    ctx.globalCompositeOperation = 'lighter'

    // Draw each LED using RGB stamps without save/restore overhead
    ledPositions.positions.forEach((position, i) => {
      const [x, y] = position

      // Scale coordinates to canvas size
      const canvasX = x * scaleX
      const canvasY = y * scaleY

      // Determine LED color and skip dark LEDs
      if (previewData?.has_frame && previewData?.frame_data) {
        const frameData = previewData.frame_data  // Uint8Array with flat RGB data

        // Access RGB values from flat array (i*3, i*3+1, i*3+2)
        const offset = i * 3
        if (offset + 2 < frameData.length) {
          const r = frameData[offset]
          const g = frameData[offset + 1]
          const b = frameData[offset + 2]

          // Skip very dim LEDs to avoid unnecessary computation (threshold: < 3)
          if (r < 3 && g < 3 && b < 3) {
            return
          }

          // Convert from linear light space to sRGB for display
          // Backend sends linear values (0-255), we need sRGB for correct display
          const rLinear = r / 255.0
          const gLinear = g / 255.0
          const bLinear = b / 255.0

          const rSrgb = linearToSrgb(rLinear)
          const gSrgb = linearToSrgb(gLinear)
          const bSrgb = linearToSrgb(bLinear)

          // Apply brightness factor for preview (values now in [0,1] sRGB space)
          const alphaR = rSrgb * previewBrightness
          const alphaG = gSrgb * previewBrightness
          const alphaB = bSrgb * previewBrightness

          const stampX = canvasX - stampR.width / 2
          const stampY = canvasY - stampR.height / 2

          // Draw each color channel with appropriate alpha (no save/restore)
          if (alphaR > 0) {
            ctx.globalAlpha = alphaR
            ctx.drawImage(stampR, stampX, stampY)
          }

          if (alphaG > 0) {
            ctx.globalAlpha = alphaG
            ctx.drawImage(stampG, stampX, stampY)
          }

          if (alphaB > 0) {
            ctx.globalAlpha = alphaB
            ctx.drawImage(stampB, stampX, stampY)
          }
        }
      } else {
        // Fallback: draw dim white LEDs when no preview data
        const stampX = canvasX - stampR.width / 2
        const stampY = canvasY - stampR.height / 2
        const dimAlpha = 0.1

        ctx.globalAlpha = dimAlpha
        ctx.drawImage(stampR, stampX, stampY)
        ctx.drawImage(stampG, stampX, stampY)
        ctx.drawImage(stampB, stampX, stampY)
      }
    })

    // Reset composite operation
    ctx.globalCompositeOperation = 'source-over'

    const renderEndTime = performance.now()
    const renderDuration = renderEndTime - renderStartTime

    // Track render durations with EWMA for logging
    const stats = renderStatsRef.current
    if (stats.ewmaRenderDuration === null) {
      stats.ewmaRenderDuration = renderDuration
      stats.renderCount = 0
      stats.lastLogTime = renderEndTime
    } else {
      stats.ewmaRenderDuration = 0.1 * renderDuration + 0.9 * stats.ewmaRenderDuration
      stats.renderCount++
    }

    // Log average render time every 5 seconds
    if (renderEndTime - stats.lastLogTime > 5000) {
      console.log(`🎨 LED render performance: ${stats.ewmaRenderDuration.toFixed(2)}ms average, ${ledPositions.positions.length} LEDs, ${stats.renderCount} frames in last 5s`)
      stats.renderCount = 0
      stats.lastLogTime = renderEndTime
    }
  }

  // Redraw LEDs when data changes
  useEffect(() => {
    // Track how often this effect runs
    if (!useEffect.lastLogTime) {
      useEffect.lastLogTime = performance.now()
      useEffect.callCount = 0
    } else {
      useEffect.callCount++
      const now = performance.now()
      if (now - useEffect.lastLogTime > 5000) {
        console.log(`🔄 useEffect triggered ${useEffect.callCount} times in last 5s (should be ~150 for 30fps)`)
        useEffect.callCount = 0
        useEffect.lastLogTime = now
      }
    }

    drawLEDs()
  }, [ledPositions, previewData, previewBrightness])

  const handlePlayPause = async () => {
    try {
      // Use renderer state instead of playlist state for play/pause logic
      const isRendererPlaying = status?.renderer_state === 'playing'
      const endpoint = isRendererPlaying ? '/api/control/pause' : '/api/control/play'
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

  const handleOptimizationIterationsChange = async (newIterations) => {
    try {
      await fetch('/api/settings/optimization-iterations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ iterations: newIterations }),
      })
      setOptimizationIterations(newIterations)
    } catch (error) {
      console.error('Failed to update optimization iterations:', error)
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
          {status?.renderer_state !== 'playing' && ledPositions && (
            <div className="absolute inset-0 flex items-center justify-center bg-dark-800 bg-opacity-50">
              <div className="text-center">
                <SpeakerXMarkIcon className="w-12 h-12 text-metal-silver mx-auto mb-2 opacity-50" />
                <p className="text-metal-silver text-sm font-mono">
                  {status?.renderer_state === 'paused' ? 'PAUSED' :
                   status?.renderer_state === 'waiting' ? 'LOADING...' :
                   (currentItem || status?.current_file) ? 'STOPPED' : 'NO CONTENT'}
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
                {/* Show playback position if available, otherwise duration */}
                {previewData && previewData.playback_position !== undefined ? (
                  <span className="text-neon-cyan">
                    {formatDuration(previewData.playback_position)}
                  </span>
                ) : currentItem?.duration ? (
                  <span>{Math.round(currentItem.duration)}s</span>
                ) : null}
                <span className={`px-2 py-1 rounded text-xs ${
                  status?.renderer_state === 'playing'
                    ? 'bg-neon-green bg-opacity-20 text-neon-green'
                    : status?.renderer_state === 'paused'
                    ? 'bg-neon-orange bg-opacity-20 text-neon-orange'
                    : status?.renderer_state === 'waiting'
                    ? 'bg-neon-cyan bg-opacity-20 text-neon-cyan'
                    : 'bg-metal-silver bg-opacity-20 text-metal-silver'
                }`}>
                  {status?.renderer_state === 'playing' ? 'PLAYING' :
                   status?.renderer_state === 'paused' ? 'PAUSED' :
                   status?.renderer_state === 'waiting' ? 'LOADING' : 'STOPPED'}
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
            aria-label={status?.renderer_state === 'playing' ? 'Pause' : 'Play'}
          >
            {status?.renderer_state === 'playing' ? (
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

        {/* Playlist position indicator - based on rendering position */}
        {playlist.items?.length > 0 && (
          <div className="mt-4 text-center">
            {(() => {
              // Use rendering_index from ControlState as the authoritative position
              const renderingIndex = status?.rendering_index >= 0 ? status.rendering_index : 0;
              return (
                <>
                  <p className="text-xs text-metal-silver font-mono">
                    {renderingIndex + 1} / {playlist.items.length}
                  </p>

                  {/* Progress bar - shows rendering position */}
                  <div className="mt-2 w-full bg-dark-700 rounded-retro h-1 overflow-hidden">
                    <div
                      className="h-full bg-neon-cyan transition-all duration-300"
                      style={{
                        width: `${((renderingIndex + 1) / playlist.items.length) * 100}%`
                      }}
                    />
                  </div>
                </>
              );
            })()}
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
              <span className="text-neon-cyan ml-2">{status.cpu_usage?.toFixed(1)}% / GPU {status.gpu_usage?.toFixed(1)}%</span>
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
            <div>
              <span className="text-metal-silver">CPU Temp:</span>
              <span className={`ml-2 ${status.cpu_temperature > 99 ? 'text-neon-red' : status.cpu_temperature > 80 ? 'text-neon-orange' : 'text-neon-cyan'}`}>
                {status.cpu_temperature?.toFixed(1) || 'N/A'}°C
              </span>
            </div>
            <div>
              <span className="text-metal-silver">GPU Temp:</span>
              <span className={`ml-2 ${status.gpu_temperature > 99 ? 'text-neon-red' : status.gpu_temperature > 80 ? 'text-neon-orange' : 'text-neon-cyan'}`}>
                {status.gpu_temperature?.toFixed(1) || 'N/A'}°C
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Preview Settings / Status */}
      <div className="retro-container">
        <h3 className="text-sm font-retro text-neon-purple mb-3">PREVIEW SETTINGS / STATUS</h3>
        <div className="space-y-4">
          {/* Brightness Control */}
          <div>
            <h4 className="text-xs font-retro text-metal-silver mb-2">BRIGHTNESS</h4>
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

          {/* Preview Frame Rate */}
          <div className="flex justify-between items-center pt-2 border-t border-dark-700">
            <span className="text-xs text-metal-silver font-mono">
              Preview Frame Rate
            </span>
            <span className="text-sm text-neon-purple font-mono">
              {previewFps > 0 ? `${previewFps.toFixed(1)} FPS` : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Optimization Iterations Control */}
      <div className="retro-container">
        <h3 className="text-sm font-retro text-neon-orange mb-3">OPTIMIZATION ITERATIONS</h3>
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <span className="text-xs text-metal-silver font-mono w-8">0</span>
            <div className="flex-1 relative">
              <input
                type="range"
                min="0"
                max="20"
                step="1"
                value={optimizationIterations}
                onChange={(e) => handleOptimizationIterationsChange(parseInt(e.target.value))}
                className="w-full h-2 bg-dark-700 rounded-retro appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right,
                    #333 0%,
                    #333 ${(optimizationIterations / 20) * 100}%,
                    #1a1a1a ${(optimizationIterations / 20) * 100}%,
                    #1a1a1a 100%)`
                }}
              />
              <div
                className="absolute top-0 h-2 bg-gradient-to-r from-neon-orange to-neon-red rounded-retro pointer-events-none"
                style={{ width: `${(optimizationIterations / 20) * 100}%` }}
              />
            </div>
            <span className="text-xs text-metal-silver font-mono w-8">20</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-metal-silver font-mono">
              {optimizationIterations === 0 ? 'Pseudo inverse only (no optimization)' : 'Number of optimization iterations for LED calculations'}
            </span>
            <span className="text-sm text-neon-orange font-mono">
              {optimizationIterations === 0 ? 'PSEUDO INV' : optimizationIterations}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HomePage
