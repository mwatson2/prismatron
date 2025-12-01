import React, { useEffect, useRef, useState, useCallback } from 'react'

// Statistics tracking for logging
let statsCounter = 0
let maxIntensity = 0
let lastLogTime = Date.now()
let cutEventCount = 0
let dropEventCount = 0

/**
 * IntensityMeter - Vertical VU-style meter for build-up intensity
 */
const IntensityMeter = ({ value }) => {
  // Clamp value to 0-1.5 range (allow overflow beyond 1.0 for emphasis)
  const clampedValue = Math.min(1.5, Math.max(0, value))
  const heightPercent = Math.min(100, clampedValue * 100)

  // Color gradient based on intensity: green -> yellow -> orange -> red
  const getGradientColor = () => {
    if (value >= 1.0) return 'from-red-500 to-red-400'
    if (value >= 0.7) return 'from-orange-500 to-yellow-400'
    if (value >= 0.4) return 'from-yellow-500 to-green-400'
    return 'from-green-500 to-green-400'
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-xs font-mono text-metal-silver">
        {Math.round(value * 100)}%
      </div>
      <div className="relative w-8 h-32 bg-dark-900 rounded-retro border border-metal-silver border-opacity-30 overflow-hidden">
        {/* Meter segments for VU-style look */}
        <div className="absolute inset-0 flex flex-col justify-end">
          <div
            className={`w-full bg-gradient-to-t ${getGradientColor()} transition-all duration-150 ease-out`}
            style={{ height: `${heightPercent}%` }}
          />
        </div>
        {/* Segment lines */}
        <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="h-px bg-dark-700" />
          ))}
        </div>
        {/* Peak marker at 100% */}
        <div className="absolute left-0 right-0 bottom-[66.67%] h-0.5 bg-red-500 opacity-50" />
      </div>
      <div className="text-xs font-mono text-neon-yellow font-bold">BUILD</div>
    </div>
  )
}

/**
 * EventLight - Circular indicator that lights up on events and fades over time
 */
const EventLight = ({ label, lastEventTime, color, fadeMs = 1000 }) => {
  const [opacity, setOpacity] = useState(0)
  const animationRef = useRef(null)

  const updateOpacity = useCallback(() => {
    if (lastEventTime <= 0) {
      setOpacity(0)
      return
    }

    const elapsed = Date.now() / 1000 - lastEventTime
    const fadeSeconds = fadeMs / 1000
    const newOpacity = Math.max(0, 1 - elapsed / fadeSeconds)
    setOpacity(newOpacity)

    if (newOpacity > 0) {
      animationRef.current = requestAnimationFrame(updateOpacity)
    }
  }, [lastEventTime, fadeMs])

  useEffect(() => {
    // Start animation when lastEventTime changes
    updateOpacity()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [updateOpacity])

  // Color configurations
  const colorConfig = {
    cyan: {
      bg: 'bg-cyan-400',
      glow: 'shadow-cyan-400',
      text: 'text-cyan-400',
      border: 'border-cyan-400',
    },
    lime: {
      bg: 'bg-lime-400',
      glow: 'shadow-lime-400',
      text: 'text-lime-400',
      border: 'border-lime-400',
    },
  }

  const cfg = colorConfig[color] || colorConfig.cyan

  return (
    <div className="flex items-center gap-3">
      {/* Indicator light */}
      <div
        className={`w-6 h-6 rounded-full border-2 ${cfg.border} transition-shadow duration-100`}
        style={{
          backgroundColor: opacity > 0 ? `rgba(${color === 'cyan' ? '34, 211, 238' : '163, 230, 53'}, ${opacity})` : 'transparent',
          boxShadow: opacity > 0 ? `0 0 ${12 * opacity}px ${4 * opacity}px rgba(${color === 'cyan' ? '34, 211, 238' : '163, 230, 53'}, ${opacity * 0.6})` : 'none',
        }}
      />
      {/* Label */}
      <div className={`text-sm font-retro font-bold ${opacity > 0.3 ? cfg.text : 'text-metal-silver'}`}>
        {label}
      </div>
    </div>
  )
}

/**
 * Build-up/Drop Detection Visualizer
 *
 * Displays real-time build-up intensity and cut/drop event indicators.
 * - Intensity meter: VU-style vertical bar showing continuous build-up progression
 * - CUT indicator: Lights up cyan when a cut event occurs (energy drop during buildup)
 * - DROP indicator: Lights up lime when a drop event occurs (bass returns after buildup/cut)
 */
const BuildDropVisualizer = ({ systemStatus }) => {
  const prevCutTimeRef = useRef(0)
  const prevDropTimeRef = useRef(0)

  useEffect(() => {
    if (!systemStatus) return

    const {
      buildup_intensity = 0.0,
      last_cut_time = 0.0,
      last_drop_time = 0.0,
    } = systemStatus

    // Track cut/drop events
    if (last_cut_time > prevCutTimeRef.current) {
      cutEventCount++
      prevCutTimeRef.current = last_cut_time
    }
    if (last_drop_time > prevDropTimeRef.current) {
      dropEventCount++
      prevDropTimeRef.current = last_drop_time
    }

    // Track maximums
    maxIntensity = Math.max(maxIntensity, buildup_intensity)
    statsCounter++

    // Log statistics every 10 seconds
    const now = Date.now()
    if (now - lastLogTime >= 10000) {
      console.log(
        `[BuildDropVisualizer] Stats (last 10s): ` +
        `updates=${statsCounter}, ` +
        `cuts=${cutEventCount}, ` +
        `drops=${dropEventCount}, ` +
        `max_intensity=${maxIntensity.toFixed(3)}`
      )

      // Reset counters
      statsCounter = 0
      cutEventCount = 0
      dropEventCount = 0
      maxIntensity = 0
      lastLogTime = now
    }
  }, [systemStatus])

  if (!systemStatus) {
    return null
  }

  const {
    buildup_intensity = 0.0,
    last_cut_time = 0.0,
    last_drop_time = 0.0,
  } = systemStatus

  return (
    <div className="retro-container">
      <h3 className="text-lg font-retro text-neon-cyan mb-4">
        BUILD-UP/DROP DETECTION
      </h3>

      {/* Main visualization: Intensity meter + Event indicators */}
      <div className="flex items-start gap-6 mb-4">
        {/* Intensity Meter */}
        <IntensityMeter value={buildup_intensity} />

        {/* Event Indicators */}
        <div className="flex flex-col gap-4 pt-6">
          <EventLight
            label="CUT"
            lastEventTime={last_cut_time}
            color="cyan"
            fadeMs={1000}
          />
          <EventLight
            label="DROP"
            lastEventTime={last_drop_time}
            color="lime"
            fadeMs={1000}
          />
        </div>
      </div>

      {/* Info footer */}
      <div className="mt-4 p-2 bg-dark-800 rounded-retro border border-metal-silver border-opacity-20">
        <div className="text-[10px] font-mono text-dark-400 text-center">
          Spectral analysis for house/trance music patterns
        </div>
      </div>
    </div>
  )
}

export default BuildDropVisualizer
