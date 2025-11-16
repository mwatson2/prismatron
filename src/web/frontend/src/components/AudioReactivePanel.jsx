import React, { useState, useEffect } from 'react'
import { SpeakerWaveIcon, PlusIcon, ArrowUpIcon, ArrowDownIcon, TrashIcon, Bars3Icon } from '@heroicons/react/24/outline'
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd'
import { useWebSocket } from '../hooks/useWebSocket'
import {
  TRIGGER_TYPES,
  EFFECT_TYPES,
  getDefaultTriggerParams,
  getDefaultEffectParams,
  shouldShowParam,
  getTemplateName,
  getTemplatePath
} from '../config/audioReactiveConfig'

const AudioReactivePanel = () => {
  // WebSocket for real-time updates
  const { audioReactiveTriggersChanged, audioConfigSaved } = useWebSocket()

  // Master enable/disable
  const [audioReactiveEnabled, setAudioReactiveEnabled] = useState(false)
  const [audioReactiveLoading, setAudioReactiveLoading] = useState(false)

  // Global settings
  const [testTriggerInterval, setTestTriggerInterval] = useState(2.0)

  // Trigger rules
  const [triggerRules, setTriggerRules] = useState([])

  // Saved indicator state
  const [showSaved, setShowSaved] = useState(false)

  // Debug: log render state
  console.log('AudioReactivePanel render - showSaved:', showSaved, 'audioConfigSaved:', audioConfigSaved)

  // Initial fetch on mount
  useEffect(() => {
    fetchAudioReactiveSettings()
  }, [])

  // Listen for WebSocket updates from other tabs/backend
  useEffect(() => {
    if (audioReactiveTriggersChanged) {
      console.log('Received audio reactive triggers update via WebSocket, refreshing...')
      fetchAudioReactiveSettings()
    }
  }, [audioReactiveTriggersChanged])

  // Listen for save notifications from WebSocket
  useEffect(() => {
    if (audioConfigSaved) {
      console.log('Audio config saved notification received, timestamp:', audioConfigSaved)
      setShowSaved(true)
      console.log('showSaved state set to true')
    }
  }, [audioConfigSaved])

  const fetchAudioReactiveSettings = async () => {
    try {
      const response = await fetch('/api/settings/audio-reactive-triggers')
      if (response.ok) {
        const data = await response.json()
        setAudioReactiveEnabled(data.enabled || false)
        setTestTriggerInterval(data.test_interval || 2.0)
        setTriggerRules(data.rules || [])
      }
    } catch (error) {
      console.error('Failed to fetch audio reactive settings:', error)
    }
  }

  const updateAudioReactiveEnabled = async (enabled) => {
    setAudioReactiveLoading(true)
    try {
      await updateServerConfig({ enabled })
      setAudioReactiveEnabled(enabled)
      console.log(`Audio reactive effects ${enabled ? 'enabled' : 'disabled'}`)
    } catch (error) {
      console.error('Failed to update audio reactive setting:', error)
    } finally {
      setAudioReactiveLoading(false)
    }
  }

  const updateServerConfig = async (updates) => {
    // Clear saved notification when making changes
    setShowSaved(false)

    const config = {
      enabled: audioReactiveEnabled,
      test_interval: testTriggerInterval,
      rules: triggerRules,
      ...updates
    }

    const response = await fetch('/api/settings/audio-reactive-triggers', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    })

    if (!response.ok) {
      throw new Error('Failed to update configuration')
    }
  }

  const updateTestInterval = async (interval) => {
    setTestTriggerInterval(interval)
    try {
      await updateServerConfig({ test_interval: interval })
    } catch (error) {
      console.error('Failed to update test interval:', error)
    }
  }

  const addRule = () => {
    const newRule = {
      id: `rule-${Date.now()}-${Math.random()}`,
      trigger: {
        type: 'beat',
        params: getDefaultTriggerParams('beat')
      },
      effect: {
        class: 'BeatBrightnessEffect',
        params: getDefaultEffectParams('BeatBrightnessEffect')
      }
    }

    const newRules = [...triggerRules, newRule]
    setTriggerRules(newRules)
    updateServerConfig({ rules: newRules })
  }

  const deleteRule = (ruleId) => {
    const newRules = triggerRules.filter(r => r.id !== ruleId)
    setTriggerRules(newRules)
    updateServerConfig({ rules: newRules })
  }

  const moveRule = (ruleId, direction) => {
    const index = triggerRules.findIndex(r => r.id === ruleId)
    if (index === -1) return

    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= triggerRules.length) return

    const newRules = [...triggerRules]
    const [rule] = newRules.splice(index, 1)
    newRules.splice(newIndex, 0, rule)

    setTriggerRules(newRules)
    updateServerConfig({ rules: newRules })
  }

  const updateRule = (ruleId, updates) => {
    const newRules = triggerRules.map(rule =>
      rule.id === ruleId ? { ...rule, ...updates } : rule
    )
    setTriggerRules(newRules)
    updateServerConfig({ rules: newRules })
  }

  const updateTriggerType = (ruleId, newType) => {
    updateRule(ruleId, {
      trigger: {
        type: newType,
        params: getDefaultTriggerParams(newType)
      }
    })
  }

  const updateTriggerParam = (ruleId, paramKey, value) => {
    const rule = triggerRules.find(r => r.id === ruleId)
    if (!rule) return

    updateRule(ruleId, {
      trigger: {
        ...rule.trigger,
        params: {
          ...rule.trigger.params,
          [paramKey]: value
        }
      }
    })
  }

  const updateEffectClass = (ruleId, newClass) => {
    updateRule(ruleId, {
      effect: {
        class: newClass,
        params: getDefaultEffectParams(newClass)
      }
    })
  }

  const updateEffectParam = (ruleId, paramKey, value) => {
    const rule = triggerRules.find(r => r.id === ruleId)
    if (!rule) return

    updateRule(ruleId, {
      effect: {
        ...rule.effect,
        params: {
          ...rule.effect.params,
          [paramKey]: value
        }
      }
    })
  }

  const onDragEnd = (result) => {
    if (!result.destination) return

    const items = Array.from(triggerRules)
    const [reorderedItem] = items.splice(result.source.index, 1)
    items.splice(result.destination.index, 0, reorderedItem)

    setTriggerRules(items)
    updateServerConfig({ rules: items })
  }

  return (
    <div className="retro-container">
      <h3 className="text-lg font-retro text-neon-pink mb-4 flex items-center justify-between">
        <div className="flex items-center">
          <SpeakerWaveIcon className="w-6 h-6 mr-2" />
          AUDIO REACTIVE EFFECTS
        </div>
        {showSaved && (
          <span className="text-sm font-mono text-neon-green px-3 py-1 bg-dark-800 rounded-retro border border-neon-green border-opacity-50 animate-pulse">
            ✓ SAVED
          </span>
        )}
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

        {/* Global Settings - Only show when enabled */}
        {audioReactiveEnabled && (
          <div className="pl-4 border-l-2 border-neon-cyan border-opacity-30 mt-4 mb-6">
            <h4 className="text-sm font-retro text-neon-orange mb-3">GLOBAL SETTINGS</h4>
            <div className="space-y-2">
              <label className="block text-xs text-metal-silver font-mono">
                TEST TRIGGER INTERVAL ({testTriggerInterval.toFixed(1)}s)
              </label>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.5"
                value={testTriggerInterval}
                onChange={(e) => updateTestInterval(parseFloat(e.target.value))}
                className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
              />
              <div className="text-xs text-metal-silver font-mono text-center">
                0.5s → {testTriggerInterval.toFixed(1)}s → 10s
              </div>
            </div>
          </div>
        )}

        {/* Trigger Rules - Only show when enabled */}
        {audioReactiveEnabled && (
          <div className="pl-4 border-l-2 border-neon-cyan border-opacity-30 mt-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-retro text-neon-purple">TRIGGER → EFFECT RULES</h4>
              <button
                onClick={addRule}
                className="retro-button px-3 py-1 text-neon-green text-xs font-retro font-bold"
              >
                <PlusIcon className="w-4 h-4 inline mr-1" />
                ADD RULE
              </button>
            </div>

            <p className="text-xs text-metal-silver font-mono mb-4">
              First matching rule wins. Drag to reorder.
            </p>

            {/* Rules List with Drag & Drop */}
            <DragDropContext onDragEnd={onDragEnd}>
              <Droppable droppableId="trigger-rules">
                {(provided) => (
                  <div
                    {...provided.droppableProps}
                    ref={provided.innerRef}
                    className="space-y-3"
                  >
                    {triggerRules.map((rule, index) => (
                      <Draggable key={rule.id} draggableId={rule.id} index={index}>
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            className={`bg-dark-800 rounded-retro border p-4 ${
                              snapshot.isDragging
                                ? 'border-neon-purple border-opacity-50 shadow-neon'
                                : 'border-metal-silver border-opacity-20'
                            }`}
                          >
                            <TriggerEffectRule
                              rule={rule}
                              index={index}
                              totalRules={triggerRules.length}
                              dragHandleProps={provided.dragHandleProps}
                              onDelete={() => deleteRule(rule.id)}
                              onMoveUp={() => moveRule(rule.id, 'up')}
                              onMoveDown={() => moveRule(rule.id, 'down')}
                              onUpdateTriggerType={(type) => updateTriggerType(rule.id, type)}
                              onUpdateTriggerParam={(key, value) => updateTriggerParam(rule.id, key, value)}
                              onUpdateEffectClass={(cls) => updateEffectClass(rule.id, cls)}
                              onUpdateEffectParam={(key, value) => updateEffectParam(rule.id, key, value)}
                            />
                          </div>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </div>
                )}
              </Droppable>
            </DragDropContext>

            {triggerRules.length === 0 && (
              <div className="text-center py-8 text-metal-silver text-sm font-mono">
                No rules configured. Click "ADD RULE" to get started.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Info Panel */}
      <div className="mt-4 p-3 bg-dark-800 rounded-retro border border-neon-purple border-opacity-20">
        <p className="text-xs text-metal-silver font-mono leading-relaxed">
          <span className="text-neon-purple">INFO:</span> Audio reactive effects create dynamic LED responses
          synchronized to music. Configure triggers (when to fire) and effects (what to do) with
          priority-based rule matching.
        </p>
      </div>
    </div>
  )
}

const TriggerEffectRule = ({
  rule,
  index,
  totalRules,
  dragHandleProps,
  onDelete,
  onMoveUp,
  onMoveDown,
  onUpdateTriggerType,
  onUpdateTriggerParam,
  onUpdateEffectClass,
  onUpdateEffectParam
}) => {
  const triggerConfig = TRIGGER_TYPES[rule.trigger.type]
  const effectConfig = EFFECT_TYPES[rule.effect.class]

  return (
    <div className="space-y-4">
      {/* Rule Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div {...dragHandleProps} className="cursor-grab active:cursor-grabbing">
            <Bars3Icon className="w-5 h-5 text-metal-silver hover:text-neon-cyan" />
          </div>
          <span className="text-sm font-retro text-neon-cyan">Rule {index + 1}</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onMoveUp}
            disabled={index === 0}
            className={`p-1 rounded ${index === 0 ? 'text-dark-600 cursor-not-allowed' : 'text-metal-silver hover:text-neon-cyan'}`}
            title="Move up"
          >
            <ArrowUpIcon className="w-4 h-4" />
          </button>
          <button
            onClick={onMoveDown}
            disabled={index === totalRules - 1}
            className={`p-1 rounded ${index === totalRules - 1 ? 'text-dark-600 cursor-not-allowed' : 'text-metal-silver hover:text-neon-cyan'}`}
            title="Move down"
          >
            <ArrowDownIcon className="w-4 h-4" />
          </button>
          <button
            onClick={onDelete}
            className="p-1 rounded text-metal-silver hover:text-neon-red"
            title="Delete rule"
          >
            <TrashIcon className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Trigger Configuration */}
      <div>
        <h5 className="text-xs font-retro text-neon-yellow mb-2">TRIGGER</h5>
        <div className="space-y-3">
          {/* Trigger Type Selector */}
          <div className="flex items-center gap-2">
            <span className="text-lg">{triggerConfig.icon}</span>
            <select
              value={rule.trigger.type}
              onChange={(e) => onUpdateTriggerType(e.target.value)}
              className="retro-input flex-1 text-sm"
            >
              {Object.entries(TRIGGER_TYPES).map(([key, config]) => (
                <option key={key} value={key}>
                  {config.name}
                </option>
              ))}
            </select>
          </div>

          <p className="text-xs text-metal-silver font-mono">{triggerConfig.description}</p>

          {/* Trigger Parameters */}
          {triggerConfig.params.map(param => (
            shouldShowParam(param, rule.trigger.params) && (
              <ParameterInput
                key={param.key}
                param={param}
                value={rule.trigger.params[param.key]}
                onChange={(value) => onUpdateTriggerParam(param.key, value)}
              />
            )
          ))}
        </div>
      </div>

      {/* Effect Configuration */}
      <div>
        <h5 className="text-xs font-retro text-neon-orange mb-2">EFFECT</h5>
        <div className="space-y-3">
          {/* Effect Class Selector */}
          <div className="flex items-center gap-2">
            <span className="text-lg">{effectConfig.icon}</span>
            <select
              value={rule.effect.class}
              onChange={(e) => onUpdateEffectClass(e.target.value)}
              className="retro-input flex-1 text-sm"
            >
              {Object.entries(EFFECT_TYPES).map(([key, config]) => (
                <option key={key} value={key}>
                  {config.name}
                </option>
              ))}
            </select>
          </div>

          <p className="text-xs text-metal-silver font-mono">{effectConfig.description}</p>

          {/* Effect Parameters */}
          {effectConfig.params.map(param => (
            shouldShowParam(param, rule.effect.params) && (
              <ParameterInput
                key={param.key}
                param={param}
                value={rule.effect.params[param.key]}
                onChange={(value) => onUpdateEffectParam(param.key, value)}
              />
            )
          ))}
        </div>
      </div>
    </div>
  )
}

const ParameterInput = ({ param, value, onChange }) => {
  const displayValue = value !== null && value !== undefined ? value : param.default

  if (param.type === 'slider') {
    // For optional sliders with null values, use middle of range for display
    const actualValue = displayValue !== null && displayValue !== undefined ? displayValue : (param.min + param.max) / 2
    const percentage = param.max <= 1 ? (actualValue * 100).toFixed(0) + '%' : actualValue.toFixed(1) + 'x'

    return (
      <div className="space-y-1">
        <label className="block text-xs text-metal-silver font-mono">
          {param.label} ({percentage})
        </label>
        <input
          type="range"
          min={param.min}
          max={param.max}
          step={param.step}
          value={actualValue}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
        />
        {param.description && (
          <p className="text-xs text-metal-silver font-mono opacity-70">{param.description}</p>
        )}
      </div>
    )
  }

  if (param.type === 'number') {
    return (
      <div className="space-y-1">
        <label className="block text-xs text-metal-silver font-mono">{param.label}</label>
        <input
          type="number"
          min={param.min}
          max={param.max}
          step={param.step}
          value={displayValue || ''}
          onChange={(e) => onChange(e.target.value ? parseFloat(e.target.value) : null)}
          placeholder={param.optional ? 'Any' : ''}
          className="retro-input w-full text-sm"
        />
        {param.description && (
          <p className="text-xs text-metal-silver font-mono opacity-70">{param.description}</p>
        )}
      </div>
    )
  }

  if (param.type === 'select') {
    return (
      <div className="space-y-1">
        <label className="block text-xs text-metal-silver font-mono">{param.label}</label>
        <select
          value={displayValue}
          onChange={(e) => onChange(e.target.value)}
          className="retro-input w-full text-sm"
        >
          {param.options.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        {param.description && (
          <p className="text-xs text-metal-silver font-mono opacity-70">{param.description}</p>
        )}
      </div>
    )
  }

  if (param.type === 'template_select') {
    const templateName = getTemplateName(displayValue)

    return (
      <div className="space-y-1">
        <label className="block text-xs text-metal-silver font-mono">{param.label}</label>
        <select
          value={templateName}
          onChange={(e) => onChange(getTemplatePath(e.target.value))}
          className="retro-input w-full text-sm"
        >
          {Object.keys(param.options).map(name => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        {param.description && (
          <p className="text-xs text-metal-silver font-mono opacity-70">{param.description}</p>
        )}
      </div>
    )
  }

  if (param.type === 'checkbox') {
    const isChecked = displayValue === true || displayValue === 'true'

    return (
      <div className="space-y-1">
        <label className="flex items-center gap-2 text-xs text-metal-silver font-mono cursor-pointer">
          <input
            type="checkbox"
            checked={isChecked}
            onChange={(e) => onChange(e.target.checked)}
            className="w-4 h-4 rounded border-2 border-metal-silver bg-dark-700 checked:bg-neon-cyan checked:border-neon-cyan focus:ring-2 focus:ring-neon-cyan focus:ring-opacity-50"
          />
          <span>{param.label}</span>
        </label>
        {param.description && (
          <p className="text-xs text-metal-silver font-mono opacity-70 ml-6">{param.description}</p>
        )}
      </div>
    )
  }

  return null
}

export default AudioReactivePanel
