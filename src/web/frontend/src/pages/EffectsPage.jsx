import React, { useState, useEffect } from 'react'
import { PlusIcon, AdjustmentsHorizontalIcon, SpeakerWaveIcon } from '@heroicons/react/24/outline'

const EffectsPage = () => {
  const [effects, setEffects] = useState([])
  const [systemFonts, setSystemFonts] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [showConfig, setShowConfig] = useState(null)
  const [customConfig, setCustomConfig] = useState({})

  // Audio reactive state
  const [audioReactiveEnabled, setAudioReactiveEnabled] = useState(false)
  const [positionShiftingEnabled, setPositionShiftingEnabled] = useState(false)
  const [maxShiftDistance, setMaxShiftDistance] = useState(3)
  const [shiftDirection, setShiftDirection] = useState('alternating')
  const [audioReactiveLoading, setAudioReactiveLoading] = useState(false)

  // Beat brightness boost state
  const [beatBrightnessEnabled, setBeatBrightnessEnabled] = useState(true)
  const [beatBrightnessIntensity, setBeatBrightnessIntensity] = useState(0.25)
  const [beatBrightnessDuration, setBeatBrightnessDuration] = useState(0.25)

  useEffect(() => {
    fetchEffects()
    fetchSystemFonts()
    fetchAudioReactiveSettings()
  }, [])

  const fetchEffects = async () => {
    try {
      const response = await fetch('/api/effects')
      if (response.ok) {
        const data = await response.json()
        setEffects(data)
      }
    } catch (error) {
      console.error('Failed to fetch effects:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchSystemFonts = async () => {
    try {
      const response = await fetch('/api/system-fonts')
      if (response.ok) {
        const data = await response.json()
        console.log('Loaded system fonts:', data.count)
        setSystemFonts(data.fonts || [])
      }
    } catch (error) {
      console.error('Failed to fetch system fonts:', error)
    }
  }

  const fetchAudioReactiveSettings = async () => {
    try {
      const response = await fetch('/api/settings/audio-reactive')
      if (response.ok) {
        const data = await response.json()
        setAudioReactiveEnabled(data.enabled || false)
        setPositionShiftingEnabled(data.position_shifting_enabled || false)
        setMaxShiftDistance(data.max_shift_distance || 3)
        setShiftDirection(data.shift_direction || 'alternating')
        setBeatBrightnessEnabled(data.beat_brightness_enabled !== undefined ? data.beat_brightness_enabled : true)
        setBeatBrightnessIntensity(data.beat_brightness_intensity || 0.25)
        setBeatBrightnessDuration(data.beat_brightness_duration || 0.25)
      }
    } catch (error) {
      console.error('Failed to fetch audio reactive settings:', error)
    }
  }

  const updateAudioReactiveEnabled = async (enabled) => {
    setAudioReactiveLoading(true)
    try {
      const response = await fetch('/api/settings/audio-reactive', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled })
      })

      if (response.ok) {
        setAudioReactiveEnabled(enabled)
        console.log(`Audio reactive effects ${enabled ? 'enabled' : 'disabled'}`)
      }
    } catch (error) {
      console.error('Failed to update audio reactive setting:', error)
    } finally {
      setAudioReactiveLoading(false)
    }
  }

  const updatePositionShifting = async (settings) => {
    try {
      const response = await fetch('/api/settings/position-shifting', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
      })

      if (response.ok) {
        const data = await response.json()
        setPositionShiftingEnabled(data.enabled)
        setMaxShiftDistance(data.max_shift_distance)
        setShiftDirection(data.shift_direction)
        console.log('Position shifting settings updated')
      }
    } catch (error) {
      console.error('Failed to update position shifting settings:', error)
    }
  }

  const updateBeatBrightness = async (settings) => {
    try {
      const response = await fetch('/api/settings/beat-brightness', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
      })

      if (response.ok) {
        const data = await response.json()
        setBeatBrightnessEnabled(data.enabled)
        setBeatBrightnessIntensity(data.intensity)
        setBeatBrightnessDuration(data.duration)
        console.log('Beat brightness settings updated')
      }
    } catch (error) {
      console.error('Failed to update beat brightness settings:', error)
    }
  }

  const addEffectToPlaylist = async (effectId, customName = null, duration = 30, config = {}) => {
    try {
      const response = await fetch(`/api/effects/${effectId}/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: customName,
          duration,
          config
        })
      })

      if (response.ok) {
        // Show success feedback
        const effectName = customName || effects.find(e => e.id === effectId)?.name
        console.log(`Added ${effectName} to playlist`)
      }
    } catch (error) {
      console.error('Failed to add effect to playlist:', error)
    }
  }

  const categories = ['all', ...new Set(effects.map(e => e.category))]
  const filteredEffects = selectedCategory === 'all'
    ? effects
    : effects.filter(e => e.category === selectedCategory)

  const handleConfigChange = (key, value) => {
    setCustomConfig(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const handleAddWithConfig = (effect) => {
    const config = { ...effect.config, ...customConfig }
    // Use custom duration from config, or default to 30
    const duration = customConfig.duration || 30
    // For text effects, use the text content as the name
    const customName = effect.id === 'text_display' && config.text
      ? config.text
      : null
    addEffectToPlaylist(effect.id, customName, duration, config)
    setShowConfig(null)
    setCustomConfig({})
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="retro-spinner w-8 h-8" />
        <span className="ml-3 text-neon-cyan font-mono">LOADING EFFECTS...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-2xl font-retro font-bold text-neon-cyan text-neon">
          VISUAL EFFECTS
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          ADD GENERATED EFFECTS TO PLAYLIST
        </p>
      </div>

      {/* Category Filter */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-pink mb-4">CATEGORIES</h3>

        <div className="flex flex-wrap gap-2">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-retro text-xs font-retro font-bold uppercase transition-all duration-200 ${
                selectedCategory === category
                  ? 'bg-neon-purple bg-opacity-20 text-neon-purple text-neon border border-neon-purple border-opacity-50'
                  : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30 hover:text-neon-purple hover:border-neon-purple hover:border-opacity-50'
              }`}
            >
              {category.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Audio Reactive Effects */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-pink mb-4 flex items-center">
          <SpeakerWaveIcon className="w-6 h-6 mr-2" />
          AUDIO REACTIVE EFFECTS
        </h3>

        {/* Master Enable/Disable */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-base font-retro text-neon-cyan">AUDIO REACTIVE MODE</h4>
              <p className="text-xs text-metal-silver font-mono">
                Enable real-time beat detection and audio-reactive effects
              </p>
            </div>
            <button
              onClick={() => updateAudioReactiveEnabled(!audioReactiveEnabled)}
              disabled={audioReactiveLoading}
              className={`px-6 py-2 rounded-retro text-sm font-retro font-bold transition-all duration-200 ${
                audioReactiveEnabled
                  ? 'bg-neon-green bg-opacity-20 text-neon-green border border-neon-green border-opacity-50'
                  : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30 hover:text-neon-green hover:border-neon-green hover:border-opacity-50'
              } ${audioReactiveLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {audioReactiveLoading ? 'UPDATING...' : audioReactiveEnabled ? 'ENABLED' : 'DISABLED'}
            </button>
          </div>

          {/* Position Shifting Controls - Only show when audio reactive is enabled */}
          {audioReactiveEnabled && (
            <div className="pl-4 border-l-2 border-neon-cyan border-opacity-30">
              <h4 className="text-sm font-retro text-neon-orange mb-3">LED POSITION SHIFTING</h4>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Enable Position Shifting */}
                <div className="space-y-2">
                  <label className="block text-xs text-metal-silver font-mono">ENABLE SHIFTING</label>
                  <button
                    onClick={() => updatePositionShifting({
                      enabled: !positionShiftingEnabled,
                      max_shift_distance: maxShiftDistance,
                      shift_direction: shiftDirection
                    })}
                    className={`w-full px-4 py-2 rounded-retro text-xs font-retro font-bold transition-all duration-200 ${
                      positionShiftingEnabled
                        ? 'bg-neon-yellow bg-opacity-20 text-neon-yellow border border-neon-yellow border-opacity-50'
                        : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30 hover:text-neon-yellow hover:border-neon-yellow hover:border-opacity-50'
                    }`}
                  >
                    {positionShiftingEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Max Shift Distance */}
                <div className="space-y-2">
                  <label className="block text-xs text-metal-silver font-mono">MAX SHIFT DISTANCE</label>
                  <select
                    value={maxShiftDistance}
                    onChange={(e) => {
                      const newDistance = parseInt(e.target.value)
                      setMaxShiftDistance(newDistance)
                      updatePositionShifting({
                        enabled: positionShiftingEnabled,
                        max_shift_distance: newDistance,
                        shift_direction: shiftDirection
                      })
                    }}
                    className="retro-input w-full text-xs"
                    disabled={!positionShiftingEnabled}
                  >
                    <option value={1}>1 Position</option>
                    <option value={2}>2 Positions</option>
                    <option value={3}>3 Positions</option>
                    <option value={4}>4 Positions</option>
                    <option value={5}>5 Positions</option>
                  </select>
                </div>

                {/* Shift Direction */}
                <div className="space-y-2">
                  <label className="block text-xs text-metal-silver font-mono">SHIFT DIRECTION</label>
                  <select
                    value={shiftDirection}
                    onChange={(e) => {
                      const newDirection = e.target.value
                      setShiftDirection(newDirection)
                      updatePositionShifting({
                        enabled: positionShiftingEnabled,
                        max_shift_distance: maxShiftDistance,
                        shift_direction: newDirection
                      })
                    }}
                    className="retro-input w-full text-xs"
                    disabled={!positionShiftingEnabled}
                  >
                    <option value="left">Left</option>
                    <option value="right">Right</option>
                    <option value="alternating">Alternating</option>
                  </select>
                </div>
              </div>

              <div className="mt-3 p-3 bg-dark-800 rounded-retro border border-neon-purple border-opacity-20">
                <p className="text-xs text-metal-silver font-mono leading-relaxed">
                  <span className="text-neon-purple">INFO:</span> Position shifting creates dynamic effects by moving LED mappings in response to beat detection.
                  Higher distances create more dramatic shifts but may affect content recognition.
                </p>
              </div>
            </div>
          )}

          {/* Beat Brightness Controls - Only show when audio reactive is enabled */}
          {audioReactiveEnabled && (
            <div className="pl-4 border-l-2 border-neon-cyan border-opacity-30 mt-4">
              <h4 className="text-sm font-retro text-neon-orange mb-3">BEAT BRIGHTNESS BOOST</h4>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Enable Beat Brightness */}
                <div className="space-y-2">
                  <label className="block text-xs text-metal-silver font-mono">ENABLE BOOST</label>
                  <button
                    onClick={() => updateBeatBrightness({
                      enabled: !beatBrightnessEnabled,
                      intensity: beatBrightnessIntensity,
                      duration: beatBrightnessDuration
                    })}
                    className={`w-full px-4 py-2 rounded-retro text-xs font-retro font-bold transition-all duration-200 ${
                      beatBrightnessEnabled
                        ? 'bg-neon-yellow bg-opacity-20 text-neon-yellow border border-neon-yellow border-opacity-50'
                        : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30 hover:text-neon-yellow hover:border-neon-yellow hover:border-opacity-50'
                    }`}
                  >
                    {beatBrightnessEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Brightness Intensity */}
                <div className="space-y-2">
                  <label className="block text-xs text-metal-silver font-mono">
                    EFFECT STRENGTH ({beatBrightnessIntensity.toFixed(1)}x)
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="5"
                    step="0.1"
                    value={beatBrightnessIntensity}
                    onChange={(e) => {
                      const newIntensity = parseFloat(e.target.value)
                      setBeatBrightnessIntensity(newIntensity)
                      updateBeatBrightness({
                        enabled: beatBrightnessEnabled,
                        intensity: newIntensity,
                        duration: beatBrightnessDuration
                      })
                    }}
                    className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                    disabled={!beatBrightnessEnabled}
                  />
                  <div className="text-xs text-metal-silver font-mono text-center">
                    0x → {beatBrightnessIntensity.toFixed(1)}x → 5x
                  </div>
                </div>

                {/* Beat Duration */}
                <div className="space-y-2">
                  <label className="block text-xs text-metal-silver font-mono">
                    DURATION ({Math.round(beatBrightnessDuration * 100)}% of beat)
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={beatBrightnessDuration}
                    onChange={(e) => {
                      const newDuration = parseFloat(e.target.value)
                      setBeatBrightnessDuration(newDuration)
                      updateBeatBrightness({
                        enabled: beatBrightnessEnabled,
                        intensity: beatBrightnessIntensity,
                        duration: newDuration
                      })
                    }}
                    className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                    disabled={!beatBrightnessEnabled}
                  />
                  <div className="text-xs text-metal-silver font-mono text-center">
                    Quick → {Math.round(beatBrightnessDuration * 100)}% → Full Beat
                  </div>
                </div>
              </div>

              <div className="mt-3 p-3 bg-dark-800 rounded-retro border border-neon-purple border-opacity-20">
                <p className="text-xs text-metal-silver font-mono leading-relaxed">
                  <span className="text-neon-purple">INFO:</span> Beat brightness boost creates a pulsing effect that brightens the entire panel on each detected beat.
                  Effect strength (0-5x) is multiplied by beat intensity and confidence for dynamic response. Duration controls pulse length.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Effects Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredEffects.map(effect => (
          <div key={effect.id} className="effect-card">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <span className="text-2xl">{effect.icon}</span>
                <div>
                  <h4 className="text-lg font-retro text-neon-cyan">{effect.name}</h4>
                  <span className="text-xs text-metal-silver font-mono uppercase">
                    {effect.category}
                  </span>
                </div>
              </div>
            </div>

            <p className="text-sm text-metal-silver mb-4 leading-relaxed">
              {effect.description}
            </p>

            {/* Effect Parameters Preview */}
            {Object.keys(effect.config).length > 0 && (
              <div className="mb-4">
                <h5 className="text-xs font-retro text-neon-orange mb-2">PARAMETERS</h5>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  {Object.entries(effect.config).slice(0, 4).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-metal-silver">{key}:</span>
                      <span className="text-neon-cyan">
                        {typeof value === 'number' ? value.toFixed(1) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-2">
              <button
                onClick={() => {
                  // For text effects, use the default text as the name
                  const customName = effect.id === 'text_display' && effect.config.text
                    ? effect.config.text
                    : null
                  addEffectToPlaylist(effect.id, customName)
                }}
                className="flex-1 retro-button px-4 py-2 text-neon-green text-sm font-retro font-bold"
              >
                <PlusIcon className="w-4 h-4 inline mr-2" />
                ADD TO PLAYLIST
              </button>

              <button
                onClick={() => setShowConfig(showConfig === effect.id ? null : effect.id)}
                className="retro-button px-3 py-2 text-neon-yellow"
                aria-label="Configure effect"
              >
                <AdjustmentsHorizontalIcon className="w-4 h-4" />
              </button>
            </div>

            {/* Configuration Panel */}
            {showConfig === effect.id && (
              <div className="mt-4 p-4 bg-dark-800 rounded-retro border border-neon-yellow border-opacity-30">
                <h5 className="text-sm font-retro text-neon-yellow mb-3">CUSTOMIZE PARAMETERS</h5>

                {effect.id === 'text_display' ? (
                  /* Text Effect Configuration */
                  <div className="space-y-4">
                    {/* Text Content */}
                    <div className="space-y-1">
                      <label className="text-xs text-metal-silver font-mono">TEXT CONTENT</label>
                      <textarea
                        defaultValue={effect.config.text}
                        onChange={(e) => handleConfigChange('text', e.target.value)}
                        className="retro-input w-full text-sm h-20 resize-none"
                        placeholder="Enter your text..."
                      />
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      {/* Font Selection */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">FONT FAMILY</label>
                        <select
                          defaultValue={effect.config.font_family}
                          onChange={(e) => handleConfigChange('font_family', e.target.value)}
                          className="retro-input w-full text-sm"
                        >
                          {systemFonts.length > 0 ? (
                            systemFonts.map(font => (
                              <option key={font.name} value={font.name}>
                                {font.name}
                              </option>
                            ))
                          ) : (
                            <>
                              <option value="arial">Arial</option>
                              <option value="helvetica">Helvetica</option>
                              <option value="times">Times New Roman</option>
                              <option value="courier">Courier</option>
                              <option value="dejavu_sans">DejaVu Sans</option>
                            </>
                          )}
                        </select>
                      </div>

                      {/* Font Style */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">FONT STYLE</label>
                        <select
                          defaultValue={effect.config.font_style || 'normal'}
                          onChange={(e) => handleConfigChange('font_style', e.target.value)}
                          className="retro-input w-full text-sm"
                        >
                          <option value="normal">Normal</option>
                          <option value="bold">Bold</option>
                          <option value="italic">Italic</option>
                          <option value="bold-italic">Bold Italic</option>
                        </select>
                      </div>

                      {/* Font Size */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">FONT SIZE</label>
                        <select
                          defaultValue={effect.config.font_size === 'auto' ? 'auto' : effect.config.font_size.toString()}
                          onChange={(e) => handleConfigChange('font_size', e.target.value === 'auto' ? 'auto' : parseInt(e.target.value))}
                          className="retro-input w-full text-sm"
                        >
                          <option value="auto">Auto (Fill Frame)</option>
                          <option value="8">8px</option>
                          <option value="10">10px</option>
                          <option value="12">12px</option>
                          <option value="14">14px</option>
                          <option value="16">16px</option>
                          <option value="18">18px</option>
                          <option value="20">20px</option>
                          <option value="24">24px</option>
                          <option value="28">28px</option>
                          <option value="32">32px</option>
                          <option value="36">36px</option>
                          <option value="42">42px</option>
                          <option value="48">48px</option>
                          <option value="56">56px</option>
                          <option value="64">64px</option>
                          <option value="72">72px</option>
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      {/* Foreground Color */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">TEXT COLOR</label>
                        <input
                          type="color"
                          defaultValue={effect.config.fg_color}
                          onChange={(e) => handleConfigChange('fg_color', e.target.value)}
                          className="w-full h-10 rounded border border-neon-cyan border-opacity-30 bg-dark-700 cursor-pointer"
                        />
                      </div>

                      {/* Background Color */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">BACKGROUND COLOR</label>
                        <input
                          type="color"
                          defaultValue={effect.config.bg_color}
                          onChange={(e) => handleConfigChange('bg_color', e.target.value)}
                          className="w-full h-10 rounded border border-neon-cyan border-opacity-30 bg-dark-700 cursor-pointer"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      {/* Animation Type */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">ANIMATION</label>
                        <select
                          defaultValue={effect.config.animation}
                          onChange={(e) => handleConfigChange('animation', e.target.value)}
                          className="retro-input w-full text-sm"
                        >
                          <option value="static">Static</option>
                          <option value="scroll">Scroll</option>
                          <option value="fade">Fade</option>
                        </select>
                      </div>

                      {/* Text Alignment */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">HORIZONTAL</label>
                        <select
                          defaultValue={effect.config.alignment}
                          onChange={(e) => handleConfigChange('alignment', e.target.value)}
                          className="retro-input w-full text-sm"
                        >
                          <option value="left">Left</option>
                          <option value="center">Center</option>
                          <option value="right">Right</option>
                        </select>
                      </div>

                      {/* Vertical Alignment */}
                      <div className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">VERTICAL</label>
                        <select
                          defaultValue={effect.config.vertical_alignment}
                          onChange={(e) => handleConfigChange('vertical_alignment', e.target.value)}
                          className="retro-input w-full text-sm"
                        >
                          <option value="top">Top</option>
                          <option value="center">Center</option>
                          <option value="bottom">Bottom</option>
                        </select>
                      </div>
                    </div>

                    {/* Duration */}
                    <div className="space-y-1">
                      <label className="text-xs text-metal-silver font-mono">DURATION (SECONDS)</label>
                      <input
                        type="number"
                        min="1"
                        max="300"
                        step="0.5"
                        defaultValue={effect.config.duration}
                        onChange={(e) => handleConfigChange('duration', parseFloat(e.target.value))}
                        className="retro-input w-full text-sm"
                      />
                    </div>

                    <div className="flex gap-2 pt-2">
                      <button
                        onClick={() => handleAddWithConfig(effect)}
                        className="flex-1 retro-button px-4 py-2 text-neon-green text-sm font-retro font-bold"
                      >
                        ADD TEXT TO PLAYLIST
                      </button>
                      <button
                        onClick={() => {
                          setShowConfig(null)
                          setCustomConfig({})
                        }}
                        className="px-4 py-2 text-metal-silver text-sm font-mono hover:text-neon-cyan"
                      >
                        CANCEL
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Generic Effect Configuration */
                  <div className="space-y-3">
                    {Object.entries(effect.config).map(([key, defaultValue]) => (
                      <div key={key} className="space-y-1">
                        <label className="text-xs text-metal-silver font-mono">
                          {key.replace('_', ' ').toUpperCase()}
                        </label>

                        {typeof defaultValue === 'number' ? (
                          <input
                            type="number"
                            step="0.1"
                            defaultValue={defaultValue}
                            onChange={(e) => handleConfigChange(key, parseFloat(e.target.value))}
                            className="retro-input w-full text-sm"
                          />
                        ) : typeof defaultValue === 'boolean' ? (
                          <label className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              defaultChecked={defaultValue}
                              onChange={(e) => handleConfigChange(key, e.target.checked)}
                              className="w-4 h-4 rounded border-neon-cyan border-opacity-30 bg-dark-700 text-neon-cyan focus:ring-neon-cyan"
                            />
                            <span className="text-xs text-metal-silver">Enabled</span>
                          </label>
                        ) : (
                          <input
                            type="text"
                            defaultValue={defaultValue}
                            onChange={(e) => handleConfigChange(key, e.target.value)}
                            className="retro-input w-full text-sm"
                          />
                        )}
                      </div>
                    ))}

                    {/* Duration Configuration */}
                    <div className="space-y-1">
                      <label className="text-xs text-metal-silver font-mono">DURATION (SECONDS)</label>
                      <input
                        type="number"
                        min="1"
                        max="300"
                        step="0.5"
                        defaultValue={30}
                        onChange={(e) => handleConfigChange('duration', parseFloat(e.target.value))}
                        className="retro-input w-full text-sm"
                      />
                    </div>

                    <div className="flex gap-2 pt-2">
                      <button
                        onClick={() => handleAddWithConfig(effect)}
                        className="flex-1 retro-button px-4 py-2 text-neon-green text-sm font-retro font-bold"
                      >
                        ADD WITH CUSTOM CONFIG
                      </button>
                      <button
                        onClick={() => {
                          setShowConfig(null)
                          setCustomConfig({})
                        }}
                        className="px-4 py-2 text-metal-silver text-sm font-mono hover:text-neon-cyan"
                      >
                        CANCEL
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredEffects.length === 0 && (
        <div className="text-center py-8">
          <p className="text-metal-silver font-mono">
            No effects found in this category
          </p>
        </div>
      )}

      {/* Effects Info */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-green mb-4">EFFECT INFORMATION</h3>

        <div className="space-y-3 text-sm font-mono text-metal-silver">
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Effects are real-time generated visual patterns</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Each effect has customizable parameters</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Default duration is 30 seconds (adjustable in playlist)</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Effects can be mixed with uploaded content</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EffectsPage
