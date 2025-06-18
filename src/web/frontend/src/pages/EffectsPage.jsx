import React, { useState, useEffect } from 'react'
import { PlusIcon, AdjustmentsHorizontalIcon } from '@heroicons/react/24/outline'

const EffectsPage = () => {
  const [effects, setEffects] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [showConfig, setShowConfig] = useState(null)
  const [customConfig, setCustomConfig] = useState({})

  useEffect(() => {
    fetchEffects()
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
    addEffectToPlaylist(effect.id, null, 30, config)
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
                onClick={() => addEffectToPlaylist(effect.id)}
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