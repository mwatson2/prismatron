import React, { useEffect, useRef, useState, useCallback } from 'react'

/**
 * AudioMeter - Compact vertical VU-style meter for audio metrics
 * Optionally tracks and displays a decaying peak indicator
 */
const AudioMeter = ({
  value,
  label,
  color = 'cyan',
  maxValue = 1.0,
  unit = '%',
  trackPeak = false,
  peakDecayMs = 2000,
}) => {
  // Peak tracking state
  const [peakValue, setPeakValue] = useState(0)
  const peakTimeRef = useRef(0)
  const peakValueRef = useRef(0) // Store peak value at time of setting
  const animationRef = useRef(null)

  // Clamp and normalize value
  const normalizedValue = Math.min(1.5, Math.max(0, value / maxValue))
  const heightPercent = Math.min(100, normalizedValue * 100)

  // Peak animation - decays linearly from stored peak value
  const animatePeak = useCallback(() => {
    const elapsed = Date.now() - peakTimeRef.current
    const progress = Math.min(1, elapsed / peakDecayMs)
    const decayedPeak = peakValueRef.current * (1 - progress)

    setPeakValue(decayedPeak > 0.001 ? decayedPeak : 0)

    if (progress < 1) {
      animationRef.current = requestAnimationFrame(animatePeak)
    }
  }, [peakDecayMs])

  // Track peak values
  useEffect(() => {
    if (!trackPeak) return

    const normalizedVal = value / maxValue
    if (normalizedVal > peakValue) {
      // New peak detected - store and start decay
      setPeakValue(normalizedVal)
      peakValueRef.current = normalizedVal
      peakTimeRef.current = Date.now()

      // Start/restart decay animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      animationRef.current = requestAnimationFrame(animatePeak)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [value, maxValue, trackPeak, peakValue, animatePeak])

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

  // Calculate peak line position (from bottom, as percentage)
  const peakHeightPercent = Math.min(100, peakValue * 100)

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
        {/* Decaying peak indicator line */}
        {trackPeak && peakValue > 0.01 && (
          <div
            className="absolute left-0 right-0 h-0.5 bg-white transition-all duration-100"
            style={{
              bottom: `${peakHeightPercent}%`,
              opacity: Math.max(0.3, peakValue),
            }}
          />
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
 * Also tracks peak values with a separate decay
 */
const DecayingMeter = ({
  value,
  lastEventTime,
  label,
  color = 'cyan',
  decayMs = 250,
  trackPeak = false,
  peakDecayMs = 2000,
}) => {
  const [displayValue, setDisplayValue] = useState(0)
  const [peakValue, setPeakValue] = useState(0)
  const lastEventTimeRef = useRef(0)
  const animationRef = useRef(null)
  const targetValueRef = useRef(0)
  const startTimeRef = useRef(0)
  const peakTimeRef = useRef(0)
  const peakValueRef = useRef(0) // Store peak value at time of setting
  const displayValueRef = useRef(0) // Track display value in ref to avoid race

  // Store config in refs to avoid recreating animate callback
  const decayMsRef = useRef(decayMs)
  const trackPeakRef = useRef(trackPeak)
  const peakDecayMsRef = useRef(peakDecayMs)
  decayMsRef.current = decayMs
  trackPeakRef.current = trackPeak
  peakDecayMsRef.current = peakDecayMs

  // Animation function that reads from refs to avoid dependency issues
  const animate = useCallback(() => {
    const now = Date.now()

    // Decay main value linearly over decayMs
    const elapsed = now - startTimeRef.current
    const progress = Math.min(1, elapsed / decayMsRef.current)
    const newValue = targetValueRef.current * (1 - progress)

    // Only update if value actually changed significantly
    if (Math.abs(newValue - displayValueRef.current) > 0.001) {
      displayValueRef.current = newValue
      setDisplayValue(newValue)
    }

    // Decay peak value linearly over peakDecayMs
    let peakStillDecaying = false
    if (trackPeakRef.current && peakTimeRef.current > 0) {
      const peakElapsed = now - peakTimeRef.current
      const peakProgress = Math.min(1, peakElapsed / peakDecayMsRef.current)
      const decayedPeak = peakValueRef.current * (1 - peakProgress)
      setPeakValue(decayedPeak > 0.001 ? decayedPeak : 0)
      peakStillDecaying = peakProgress < 1
    }

    if (progress < 1 || peakStillDecaying) {
      animationRef.current = requestAnimationFrame(animate)
    }
  }, []) // No dependencies - reads from refs

  // Handle new events
  useEffect(() => {
    // Check if this is a new event
    if (lastEventTime > lastEventTimeRef.current) {
      lastEventTimeRef.current = lastEventTime
      targetValueRef.current = value
      startTimeRef.current = Date.now()

      // Update peak if new value is higher than current decayed peak
      // Use ref to get current peak value to avoid stale closure
      if (trackPeak && value > peakValueRef.current) {
        setPeakValue(value)
        peakValueRef.current = value // Store for linear decay calculation
        peakTimeRef.current = Date.now()
      }

      // Cancel any existing animation and start new decay
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }

      // Set initial value immediately via ref and state
      displayValueRef.current = value
      setDisplayValue(value)

      // Delay animation start by one frame to ensure state is committed
      // This prevents the first animate() call from racing with setDisplayValue
      animationRef.current = requestAnimationFrame(() => {
        animationRef.current = requestAnimationFrame(animate)
      })
    }
  }, [lastEventTime, value, animate, trackPeak]) // Removed peakValue dependency

  // Cleanup on unmount only
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  // Calculate peak line position
  const peakHeightPercent = Math.min(100, peakValue * 100)

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-[10px] font-mono text-metal-silver whitespace-nowrap">
        {`${Math.round(displayValue * 100)}%`}
      </div>
      <div className="relative w-6 h-24 bg-dark-900 rounded-retro border border-metal-silver border-opacity-30 overflow-hidden">
        {/* Meter fill */}
        <div className="absolute inset-0 flex flex-col justify-end">
          <div
            className={`w-full bg-gradient-to-t from-${color}-500 to-${color}-400 transition-all duration-75 ease-out`}
            style={{ height: `${Math.min(100, displayValue * 100)}%` }}
          />
        </div>
        {/* Segment lines */}
        <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-px bg-dark-700" />
          ))}
        </div>
        {/* Decaying peak indicator line */}
        {trackPeak && peakValue > 0.01 && (
          <div
            className="absolute left-0 right-0 h-0.5 bg-white transition-all duration-100"
            style={{
              bottom: `${peakHeightPercent}%`,
              opacity: Math.max(0.3, peakValue),
            }}
          />
        )}
      </div>
      <div className="text-[10px] font-mono text-metal-silver font-bold text-center leading-tight max-w-[48px]">
        {label}
      </div>
    </div>
  )
}

/**
 * BPMSpeedometer - Speedometer-style BPM display with arc gauge
 */
const BPMSpeedometer = ({ bpm, minBpm = 60, maxBpm = 180 }) => {
  // Clamp BPM to range
  const clampedBpm = Math.max(minBpm, Math.min(maxBpm, bpm))

  // Calculate angle (-180 = left, 0 = right for a semi-circle)
  const normalizedValue = (clampedBpm - minBpm) / (maxBpm - minBpm)
  const angle = -180 + (normalizedValue * 180) // -180 to 0 degrees (matches arc)

  // SVG dimensions
  const size = 80
  const strokeWidth = 6
  const radius = (size - strokeWidth) / 2
  const center = size / 2

  // Arc path for background (semi-circle from left to right)
  const createArc = (startAngle, endAngle) => {
    const startRad = (startAngle * Math.PI) / 180
    const endRad = (endAngle * Math.PI) / 180
    const x1 = center + radius * Math.cos(startRad)
    const y1 = center + radius * Math.sin(startRad)
    const x2 = center + radius * Math.cos(endRad)
    const y2 = center + radius * Math.sin(endRad)
    const largeArc = endAngle - startAngle > 180 ? 1 : 0
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`
  }

  // Needle endpoint
  const needleLength = radius - 8
  const needleAngle = (angle * Math.PI) / 180
  const needleX = center + needleLength * Math.cos(needleAngle)
  const needleY = center + needleLength * Math.sin(needleAngle)

  // Color based on BPM (slower = cooler, faster = warmer)
  const getColor = () => {
    if (bpm < 90) return '#22d3ee' // cyan
    if (bpm < 120) return '#a3e635' // lime
    if (bpm < 140) return '#facc15' // yellow
    if (bpm < 160) return '#fb923c' // orange
    return '#f87171' // red
  }

  return (
    <div className="flex flex-col items-center">
      {/* Speedometer arc */}
      <svg width={size} height={size / 2} viewBox={`0 0 ${size} ${size / 2}`}>
        {/* Background arc */}
        <path
          d={createArc(-180, 0)}
          fill="none"
          stroke="#333"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Colored progress arc */}
        <path
          d={createArc(-180, -180 + normalizedValue * 180)}
          fill="none"
          stroke={getColor()}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${getColor()})` }}
        />

        {/* Tick marks */}
        {[60, 90, 120, 150, 180].map((tick) => {
          const tickNorm = (tick - minBpm) / (maxBpm - minBpm)
          const tickAngle = (-180 + tickNorm * 180) * Math.PI / 180
          const innerR = radius - strokeWidth / 2 - 4
          const outerR = radius - strokeWidth / 2 - 8
          return (
            <line
              key={tick}
              x1={center + innerR * Math.cos(tickAngle)}
              y1={center + innerR * Math.sin(tickAngle)}
              x2={center + outerR * Math.cos(tickAngle)}
              y2={center + outerR * Math.sin(tickAngle)}
              stroke="#666"
              strokeWidth={1}
            />
          )
        })}

        {/* Needle */}
        <line
          x1={center}
          y1={center}
          x2={needleX}
          y2={needleY}
          stroke={getColor()}
          strokeWidth={2}
          strokeLinecap="round"
          style={{
            filter: `drop-shadow(0 0 3px ${getColor()})`,
            transition: 'all 0.15s ease-out'
          }}
        />

        {/* Center dot */}
        <circle
          cx={center}
          cy={center}
          r={4}
          fill={getColor()}
          style={{ filter: `drop-shadow(0 0 3px ${getColor()})` }}
        />
      </svg>
      {/* BPM value - separate from arc with spacing */}
      <div
        className="text-sm font-mono font-bold mt-1"
        style={{ color: getColor(), textShadow: `0 0 4px ${getColor()}` }}
      >
        {Math.round(bpm)}
      </div>
      <div className="text-[10px] font-mono text-metal-silver font-bold -mt-0.5">
        BPM
      </div>
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
    purple: {
      rgb: '192, 132, 252',
      border: 'border-purple-400',
      text: 'text-purple-400',
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
 * - Audio Level: Raw RMS audio level (continuous, with peak)
 * - AGC Gain: Automatic Gain Control setting in dB (continuous, no peak)
 * - Beat Intensity: Intensity of detected beats (decays after beat, with peak)
 * - Beat Confidence: Confidence of beat detection (decays after beat, with peak)
 * - BuildUp Intensity: Build-up progression for drops (continuous, no peak)
 *
 * Plus event lights for CUT and DROP events.
 */
const AudioVisualizer = ({ systemStatus }) => {
  if (!systemStatus) {
    return null
  }

  const {
    audio_level = 0.0,
    agc_gain_db = 0.0,
    current_bpm = 120.0,
    last_beat_time = 0.0,
    beat_intensity = 0.0,
    beat_confidence = 0.0,
    buildup_intensity = 0.0,
    last_cut_time = 0.0,
    last_drop_time = 0.0,
    last_downbeat_time = 0.0,
  } = systemStatus

  // Normalize AGC gain for display (-20dB to +20dB range -> 0 to 1)
  const normalizedAgcGain = (agc_gain_db + 20) / 40

  return (
    <div className="retro-container">
      {/* Two-column layout */}
      <div className="flex gap-4">
        {/* Left column: Title and meters */}
        <div className="flex-1">
          <h3 className="text-sm font-retro text-neon-cyan mb-3">
            AUDIO ANALYZER
          </h3>
          <div className="flex items-end justify-between gap-2">
            {/* Audio Level - continuous with peak */}
            <AudioMeter
              value={audio_level}
              label="LEVEL"
              color="green"
              maxValue={0.5}
              unit="%"
              trackPeak={true}
              peakDecayMs={2000}
            />

            {/* AGC Gain - continuous, no peak */}
            <AudioMeter
              value={normalizedAgcGain}
              label="AGC"
              color="purple"
              maxValue={1.0}
              unit="dB"
              trackPeak={false}
            />

            {/* Beat Intensity - decaying with peak */}
            <DecayingMeter
              value={beat_intensity}
              lastEventTime={last_beat_time}
              label="BEAT"
              color="yellow"
              decayMs={250}
              trackPeak={true}
              peakDecayMs={2000}
            />

            {/* Beat Confidence - decaying with peak */}
            <DecayingMeter
              value={beat_confidence}
              lastEventTime={last_beat_time}
              label="CONF"
              color="orange"
              decayMs={250}
              trackPeak={true}
              peakDecayMs={2000}
            />

            {/* Build-up intensity - continuous, no peak */}
            <AudioMeter
              value={buildup_intensity}
              label="BUILD"
              color="dynamic"
              maxValue={1.5}
              unit="%"
              trackPeak={false}
            />
          </div>
        </div>

        {/* Right column: Speedometer and event lights */}
        <div className="flex flex-col items-center justify-between pt-4">
          <BPMSpeedometer bpm={current_bpm} />
          <div className="flex gap-2">
            <EventLight
              label="DOWN"
              lastEventTime={last_downbeat_time}
              color="purple"
              fadeMs={500}
            />

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
      </div>
    </div>
  )
}

export default AudioVisualizer
