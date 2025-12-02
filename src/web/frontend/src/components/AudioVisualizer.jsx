import React, { useEffect, useRef, useState, useCallback } from 'react'

/**
 * AudioMeter - Compact vertical VU-style meter for audio metrics
 */
const AudioMeter = ({ value, label, color = 'cyan', maxValue = 1.0, showPeakLine = false, unit = '%' }) => {
  // Clamp and normalize value
  const normalizedValue = Math.min(1.5, Math.max(0, value / maxValue))
  const heightPercent = Math.min(100, normalizedValue * 100)

  // Color configurations
  const colorConfig = {
    cyan: 'from-cyan-500 to-cyan-400',
    green: 'from-green-500 to-green-400',
    yellow: 'from-yellow-500 to-yellow-400',
    orange: 'from-orange-500 to-orange-400',
    purple: 'from-purple-500 to-purple-400',
  }

  // Dynamic color based on value for some meters
  const getGradientColor = () => {
    if (color === 'dynamic') {
      if (normalizedValue >= 1.0) return 'from-red-500 to-red-400'
      if (normalizedValue >= 0.7) return 'from-orange-500 to-yellow-400'
      if (normalizedValue >= 0.4) return 'from-yellow-500 to-green-400'
      return 'from-green-500 to-green-400'
    }
    return colorConfig[color] || colorConfig.cyan
  }

  // Format display value
  const displayValue = unit === '%'
    ? `${Math.round(value * 100)}%`
    : unit === 'dB'
    ? `${value.toFixed(1)}dB`
    : value.toFixed(2)

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-[10px] font-mono text-metal-silver whitespace-nowrap">
        {displayValue}
      </div>
      <div className="relative w-6 h-24 bg-dark-900 rounded-retro border border-metal-silver border-opacity-30 overflow-hidden">
        {/* Meter fill */}
        <div className="absolute inset-0 flex flex-col justify-end">
          <div
            className={`w-full bg-gradient-to-t ${getGradientColor()} transition-all duration-75 ease-out`}
            style={{ height: `${heightPercent}%` }}
          />
        </div>
        {/* Segment lines */}
        <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-px bg-dark-700" />
          ))}
        </div>
        {/* Peak marker at 100% */}
        {showPeakLine && (
          <div className="absolute left-0 right-0 bottom-[66.67%] h-0.5 bg-red-500 opacity-50" />
        )}
      </div>
      <div className="text-[10px] font-mono text-metal-silver font-bold text-center leading-tight max-w-[48px]">
        {label}
      </div>
    </div>
  )
}

/**
 * DecayingMeter - Meter that receives values on events and decays over time
 */
const DecayingMeter = ({ value, lastEventTime, label, color = 'cyan', decayMs = 250 }) => {
  const [displayValue, setDisplayValue] = useState(0)
  const lastEventTimeRef = useRef(0)
  const animationRef = useRef(null)
  const targetValueRef = useRef(0)
  const startTimeRef = useRef(0)

  const animate = useCallback(() => {
    const elapsed = Date.now() - startTimeRef.current
    const progress = Math.min(1, elapsed / decayMs)
    const newValue = targetValueRef.current * (1 - progress)

    setDisplayValue(newValue)

    if (progress < 1) {
      animationRef.current = requestAnimationFrame(animate)
    }
  }, [decayMs])

  useEffect(() => {
    // Check if this is a new event
    if (lastEventTime > lastEventTimeRef.current) {
      lastEventTimeRef.current = lastEventTime
      targetValueRef.current = value
      startTimeRef.current = Date.now()

      // Cancel any existing animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }

      // Start decay animation
      setDisplayValue(value)
      animationRef.current = requestAnimationFrame(animate)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [lastEventTime, value, animate])

  return (
    <AudioMeter
      value={displayValue}
      label={label}
      color={color}
      maxValue={1.0}
      unit="%"
    />
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
      rgb: '34, 211, 238',
      border: 'border-cyan-400',
      text: 'text-cyan-400',
    },
    lime: {
      rgb: '163, 230, 53',
      border: 'border-lime-400',
      text: 'text-lime-400',
    },
  }

  const cfg = colorConfig[color] || colorConfig.cyan

  return (
    <div className="flex flex-col items-center gap-1">
      {/* Indicator light */}
      <div
        className={`w-8 h-8 rounded-full border-2 ${cfg.border} transition-shadow duration-100`}
        style={{
          backgroundColor: opacity > 0 ? `rgba(${cfg.rgb}, ${opacity})` : 'transparent',
          boxShadow: opacity > 0 ? `0 0 ${12 * opacity}px ${4 * opacity}px rgba(${cfg.rgb}, ${opacity * 0.6})` : 'none',
        }}
      />
      {/* Label */}
      <div className={`text-xs font-mono font-bold ${opacity > 0.3 ? cfg.text : 'text-metal-silver'}`}>
        {label}
      </div>
    </div>
  )
}

/**
 * AudioVisualizer - Comprehensive audio metrics display
 *
 * Displays 5 side-by-side meters:
 * - Audio Level: Raw RMS audio level (continuous)
 * - AGC Gain: Automatic Gain Control setting in dB (continuous)
 * - Beat Intensity: Intensity of detected beats (decays after beat)
 * - Beat Confidence: Confidence of beat detection (decays after beat)
 * - BuildUp Intensity: Build-up progression for drops (continuous)
 *
 * Plus event lights for CUT and DROP events.
 */
const AudioVisualizer = ({ systemStatus }) => {
  const prevBeatTimeRef = useRef(0)

  if (!systemStatus) {
    return null
  }

  const {
    audio_level = 0.0,
    agc_gain_db = 0.0,
    last_beat_time = 0.0,
    beat_intensity = 0.0,
    beat_confidence = 0.0,
    buildup_intensity = 0.0,
    last_cut_time = 0.0,
    last_drop_time = 0.0,
  } = systemStatus

  // Normalize AGC gain for display (-20dB to +20dB range -> 0 to 1)
  const normalizedAgcGain = (agc_gain_db + 20) / 40

  return (
    <div className="retro-container">
      <h3 className="text-sm font-retro text-neon-cyan mb-3">
        AUDIO ANALYZER
      </h3>

      {/* Main visualization: 5 meters + event lights */}
      <div className="flex items-end justify-between gap-2">
        {/* Continuous meters */}
        <AudioMeter
          value={audio_level}
          label="LEVEL"
          color="green"
          maxValue={0.5}
          unit="%"
        />

        <AudioMeter
          value={normalizedAgcGain}
          label="AGC"
          color="purple"
          maxValue={1.0}
          unit="dB"
        />

        {/* Decaying meters (triggered by beats) */}
        <DecayingMeter
          value={beat_intensity}
          lastEventTime={last_beat_time}
          label="BEAT"
          color="yellow"
          decayMs={250}
        />

        <DecayingMeter
          value={beat_confidence}
          lastEventTime={last_beat_time}
          label="CONF"
          color="orange"
          decayMs={250}
        />

        {/* Build-up intensity (continuous) */}
        <AudioMeter
          value={buildup_intensity}
          label="BUILD"
          color="dynamic"
          maxValue={1.5}
          showPeakLine={true}
          unit="%"
        />

        {/* Spacer */}
        <div className="w-2" />

        {/* Event lights */}
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
  )
}

export default AudioVisualizer
