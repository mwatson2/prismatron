import React, { useState, useEffect } from 'react'
import { SpeakerWaveIcon, PlusIcon, ArrowUpIcon, ArrowDownIcon, TrashIcon, Bars3Icon, ChevronDownIcon, ChevronRightIcon, PencilIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { useWebSocket } from '../hooks/useWebSocket'
import {
  TRIGGER_TYPES,
  EFFECT_TYPES,
  EVENT_EFFECT_TYPES,
  getDefaultTriggerParams,
  getDefaultEffectParams,
  getDefaultEventEffectParams,
  shouldShowParam,
  getTemplateName,
  getTemplatePath,
  DEFAULT_CUT_EFFECT,
  DEFAULT_DROP_EFFECT,
  DEFAULT_SPARKLE_EFFECT,
  SPARKLE_CURVE_OPTIONS
} from '../config/audioReactiveConfig'

const AudioReactivePanel = () => {
  // WebSocket for real-time updates
  const { audioReactiveTriggersChanged, audioConfigSaved } = useWebSocket()

  // Master enable/disable
  const [audioReactiveEnabled, setAudioReactiveEnabled] = useState(false)
  const [audioReactiveLoading, setAudioReactiveLoading] = useState(false)

  // Transition on downbeat setting
  const [transitionOnDownbeat, setTransitionOnDownbeat] = useState(false)
  const [transitionOnDownbeatLoading, setTransitionOnDownbeatLoading] = useState(false)

  // Global settings
  const [testTriggerInterval, setTestTriggerInterval] = useState(2.0)

  // Common rules (always active)
  const [commonRules, setCommonRules] = useState([])

  // Carousel rule sets (rotates every N beats)
  const [carouselRuleSets, setCarouselRuleSets] = useState([])
  const [carouselBeatInterval, setCarouselBeatInterval] = useState(4)

  // Cut effect configuration
  const [cutEffect, setCutEffect] = useState(DEFAULT_CUT_EFFECT)

  // Drop effect configuration
  const [dropEffect, setDropEffect] = useState(DEFAULT_DROP_EFFECT)

  // Sparkle effect configuration (buildup-triggered)
  const [sparkleEffect, setSparkleEffect] = useState(DEFAULT_SPARKLE_EFFECT)

  // Collapsed state for sections
  const [collapsedSections, setCollapsedSections] = useState({
    common: false,
    carousel: false,
    cutEffect: false,
    dropEffect: false,
    sparkleEffect: false,
  })

  // Editing state for rule set names
  const [editingSetId, setEditingSetId] = useState(null)
  const [editingSetName, setEditingSetName] = useState('')

  // Saved indicator state
  const [showSaved, setShowSaved] = useState(false)

  // Initial fetch on mount
  useEffect(() => {
    fetchAudioReactiveSettings()
    fetchTransitionOnDownbeat()
  }, [])

  const fetchTransitionOnDownbeat = async () => {
    try {
      const response = await fetch('/api/settings/transition-on-downbeat')
      if (response.ok) {
        const data = await response.json()
        setTransitionOnDownbeat(data.enabled || false)
      }
    } catch (error) {
      console.error('Failed to fetch transition-on-downbeat setting:', error)
    }
  }

  // Listen for WebSocket updates
  useEffect(() => {
    if (audioReactiveTriggersChanged) {
      fetchAudioReactiveSettings()
    }
  }, [audioReactiveTriggersChanged])

  // Listen for save notifications
  useEffect(() => {
    if (audioConfigSaved) {
      setShowSaved(true)
    }
  }, [audioConfigSaved])

  const fetchAudioReactiveSettings = async () => {
    try {
      const response = await fetch('/api/settings/audio-reactive-triggers')
      if (response.ok) {
        const data = await response.json()
        setAudioReactiveEnabled(data.enabled || false)
        setTestTriggerInterval(data.test_interval || 2.0)
        setCommonRules(data.common_rules || [])
        setCarouselRuleSets(data.carousel_rule_sets || [])
        setCarouselBeatInterval(data.carousel_beat_interval || 4)
        // Load cut/drop effect configuration
        if (data.cut_effect) {
          setCutEffect(data.cut_effect)
        }
        if (data.drop_effect) {
          setDropEffect(data.drop_effect)
        }
        // Load sparkle effect configuration
        if (data.sparkle_effect) {
          setSparkleEffect(data.sparkle_effect)
        }
      }
    } catch (error) {
      console.error('Failed to fetch audio reactive settings:', error)
    }
  }

  const updateServerConfig = async (updates) => {
    setShowSaved(false)

    const config = {
      enabled: audioReactiveEnabled,
      test_interval: testTriggerInterval,
      common_rules: commonRules,
      carousel_rule_sets: carouselRuleSets,
      carousel_beat_interval: carouselBeatInterval,
      cut_effect: cutEffect,
      drop_effect: dropEffect,
      sparkle_effect: sparkleEffect,
      ...updates
    }

    const response = await fetch('/api/settings/audio-reactive-triggers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    })

    if (!response.ok) {
      throw new Error('Failed to update configuration')
    }
  }

  const updateAudioReactiveEnabled = async (enabled) => {
    setAudioReactiveLoading(true)
    try {
      await updateServerConfig({ enabled })
      setAudioReactiveEnabled(enabled)
    } catch (error) {
      console.error('Failed to update audio reactive setting:', error)
    } finally {
      setAudioReactiveLoading(false)
    }
  }

  const updateTransitionOnDownbeat = async (enabled) => {
    setTransitionOnDownbeatLoading(true)
    try {
      const response = await fetch('/api/settings/transition-on-downbeat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      })
      if (response.ok) {
        setTransitionOnDownbeat(enabled)
      }
    } catch (error) {
      console.error('Failed to update transition-on-downbeat setting:', error)
    } finally {
      setTransitionOnDownbeatLoading(false)
    }
  }

  const toggleSection = (sectionKey) => {
    setCollapsedSections(prev => ({
      ...prev,
      [sectionKey]: !prev[sectionKey]
    }))
  }

  // === Common Rules CRUD ===
  const deleteCommonRule = (ruleId) => {
    const newRules = commonRules.filter(r => r.id !== ruleId)
    setCommonRules(newRules)
    updateServerConfig({ common_rules: newRules })
  }

  const moveCommonRule = (ruleId, direction) => {
    const index = commonRules.findIndex(r => r.id === ruleId)
    if (index === -1) return

    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= commonRules.length) return

    const newRules = [...commonRules]
    const [rule] = newRules.splice(index, 1)
    newRules.splice(newIndex, 0, rule)

    setCommonRules(newRules)
    updateServerConfig({ common_rules: newRules })
  }

  const updateCommonRule = (ruleId, updates) => {
    const newRules = commonRules.map(rule =>
      rule.id === ruleId ? { ...rule, ...updates } : rule
    )
    setCommonRules(newRules)
    updateServerConfig({ common_rules: newRules })
  }

  // === Carousel Rule Set CRUD ===
  const deleteCarouselRuleSet = (setId) => {
    const newSets = carouselRuleSets.filter(s => s.id !== setId)
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  const moveCarouselRuleSet = (setId, direction) => {
    const index = carouselRuleSets.findIndex(s => s.id === setId)
    if (index === -1) return

    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= carouselRuleSets.length) return

    const newSets = [...carouselRuleSets]
    const [set] = newSets.splice(index, 1)
    newSets.splice(newIndex, 0, set)

    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  const updateCarouselRuleSetName = (setId, newName) => {
    const newSets = carouselRuleSets.map(set =>
      set.id === setId ? { ...set, name: newName } : set
    )
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  const startEditingSetName = (setId, currentName) => {
    setEditingSetId(setId)
    setEditingSetName(currentName)
  }

  const saveSetName = (setId) => {
    if (editingSetName.trim()) {
      updateCarouselRuleSetName(setId, editingSetName.trim())
    }
    setEditingSetId(null)
    setEditingSetName('')
  }

  const cancelEditingSetName = () => {
    setEditingSetId(null)
    setEditingSetName('')
  }

  // === Carousel Rule CRUD (within a rule set) ===
  const addCarouselRule = (setId) => {
    const newRule = {
      id: `rule-${Date.now()}-${Math.random()}`,
      trigger: { type: 'beat', params: getDefaultTriggerParams('beat') },
      effect: { class: 'TemplateEffect', params: getDefaultEffectParams('TemplateEffect') }
    }

    const newSets = carouselRuleSets.map(set =>
      set.id === setId ? { ...set, rules: [...set.rules, newRule] } : set
    )
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  const deleteCarouselRule = (setId, ruleId) => {
    const newSets = carouselRuleSets.map(set =>
      set.id === setId ? { ...set, rules: set.rules.filter(r => r.id !== ruleId) } : set
    )
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  const moveCarouselRule = (setId, ruleId, direction) => {
    const set = carouselRuleSets.find(s => s.id === setId)
    if (!set) return

    const index = set.rules.findIndex(r => r.id === ruleId)
    if (index === -1) return

    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= set.rules.length) return

    const newRules = [...set.rules]
    const [rule] = newRules.splice(index, 1)
    newRules.splice(newIndex, 0, rule)

    const newSets = carouselRuleSets.map(s =>
      s.id === setId ? { ...s, rules: newRules } : s
    )
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  const updateCarouselRule = (setId, ruleId, updates) => {
    const newSets = carouselRuleSets.map(set =>
      set.id === setId
        ? { ...set, rules: set.rules.map(rule => rule.id === ruleId ? { ...rule, ...updates } : rule) }
        : set
    )
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  // === Generic Rule Update Helpers ===
  const updateTriggerType = (ruleId, newType, isCommon, setId = null) => {
    const updates = {
      trigger: {
        type: newType,
        params: getDefaultTriggerParams(newType)
      }
    }

    if (isCommon) {
      updateCommonRule(ruleId, updates)
    } else {
      updateCarouselRule(setId, ruleId, updates)
    }
  }

  const updateTriggerParam = (ruleId, paramKey, value, isCommon, setId = null) => {
    const rules = isCommon ? commonRules : carouselRuleSets.find(s => s.id === setId)?.rules || []
    const rule = rules.find(r => r.id === ruleId)
    if (!rule) return

    const updates = {
      trigger: {
        ...rule.trigger,
        params: {
          ...rule.trigger.params,
          [paramKey]: value
        }
      }
    }

    if (isCommon) {
      updateCommonRule(ruleId, updates)
    } else {
      updateCarouselRule(setId, ruleId, updates)
    }
  }

  const updateEffectClass = (ruleId, newClass, isCommon, setId = null) => {
    const updates = {
      effect: {
        class: newClass,
        params: getDefaultEffectParams(newClass)
      }
    }

    if (isCommon) {
      updateCommonRule(ruleId, updates)
    } else {
      updateCarouselRule(setId, ruleId, updates)
    }
  }

  const updateEffectParam = (ruleId, paramKey, value, isCommon, setId = null) => {
    const rules = isCommon ? commonRules : carouselRuleSets.find(s => s.id === setId)?.rules || []
    const rule = rules.find(r => r.id === ruleId)
    if (!rule) return

    const updates = {
      effect: {
        ...rule.effect,
        params: {
          ...rule.effect.params,
          [paramKey]: value
        }
      }
    }

    if (isCommon) {
      updateCommonRule(ruleId, updates)
    } else {
      updateCarouselRule(setId, ruleId, updates)
    }
  }

  // Common Rules functions
  const addCommonRule = () => {
    const newRule = {
      id: `rule-${Date.now()}-${Math.random()}`,
      trigger: { type: 'beat', params: getDefaultTriggerParams('beat') },
      effect: { class: 'BeatBrightnessEffect', params: getDefaultEffectParams('BeatBrightnessEffect') }
    }
    const newRules = [...commonRules, newRule]
    setCommonRules(newRules)
    updateServerConfig({ common_rules: newRules })
  }

  // Carousel Rule Set functions
  const addCarouselRuleSet = () => {
    const newSet = {
      id: `set-${Date.now()}-${Math.random()}`,
      name: `Rule Set ${carouselRuleSets.length + 1}`,
      rules: []
    }
    const newSets = [...carouselRuleSets, newSet]
    setCarouselRuleSets(newSets)
    updateServerConfig({ carousel_rule_sets: newSets })
  }

  // === Cut Effect Functions ===
  const updateCutEffectEnabled = (enabled) => {
    const newCutEffect = { ...cutEffect, enabled }
    setCutEffect(newCutEffect)
    updateServerConfig({ cut_effect: newCutEffect })
  }

  const updateCutEffectClass = (effect_class) => {
    const newCutEffect = {
      ...cutEffect,
      effect_class,
      params: getDefaultEventEffectParams(effect_class)
    }
    setCutEffect(newCutEffect)
    updateServerConfig({ cut_effect: newCutEffect })
  }

  const updateCutEffectParam = (key, value) => {
    const newCutEffect = {
      ...cutEffect,
      params: { ...cutEffect.params, [key]: value }
    }
    setCutEffect(newCutEffect)
    updateServerConfig({ cut_effect: newCutEffect })
  }

  // === Drop Effect Functions ===
  const updateDropEffectEnabled = (enabled) => {
    const newDropEffect = { ...dropEffect, enabled }
    setDropEffect(newDropEffect)
    updateServerConfig({ drop_effect: newDropEffect })
  }

  const updateDropEffectClass = (effect_class) => {
    const newDropEffect = {
      ...dropEffect,
      effect_class,
      params: getDefaultEventEffectParams(effect_class)
    }
    setDropEffect(newDropEffect)
    updateServerConfig({ drop_effect: newDropEffect })
  }

  const updateDropEffectParam = (key, value) => {
    const newDropEffect = {
      ...dropEffect,
      params: { ...dropEffect.params, [key]: value }
    }
    setDropEffect(newDropEffect)
    updateServerConfig({ drop_effect: newDropEffect })
  }

  // === Sparkle Effect Functions ===
  const updateSparkleEffectEnabled = (enabled) => {
    const newSparkleEffect = { ...sparkleEffect, enabled }
    setSparkleEffect(newSparkleEffect)
    updateServerConfig({ sparkle_effect: newSparkleEffect })
  }

  const updateSparkleEffectDensity = (key, value) => {
    const newSparkleEffect = {
      ...sparkleEffect,
      density: { ...sparkleEffect.density, [key]: value }
    }
    setSparkleEffect(newSparkleEffect)
    updateServerConfig({ sparkle_effect: newSparkleEffect })
  }

  const updateSparkleEffectInterval = (key, value) => {
    const newSparkleEffect = {
      ...sparkleEffect,
      interval_ms: { ...sparkleEffect.interval_ms, [key]: value }
    }
    setSparkleEffect(newSparkleEffect)
    updateServerConfig({ sparkle_effect: newSparkleEffect })
  }

  const updateSparkleEffectFadeMultiplier = (value) => {
    const newSparkleEffect = { ...sparkleEffect, fade_multiplier: value }
    setSparkleEffect(newSparkleEffect)
    updateServerConfig({ sparkle_effect: newSparkleEffect })
  }

  const updateSparkleEffectRandomColors = (enabled) => {
    const newSparkleEffect = { ...sparkleEffect, random_colors: enabled }
    setSparkleEffect(newSparkleEffect)
    updateServerConfig({ sparkle_effect: newSparkleEffect })
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

        {/* Only show configuration when enabled */}
        {audioReactiveEnabled && (
          <div className="space-y-6">
            {/* Global Settings */}
            <div className="pl-4 border-l-2 border-neon-cyan border-opacity-30">
              <h4 className="text-sm font-retro text-neon-orange mb-3">GLOBAL SETTINGS</h4>
              <div className="space-y-4">
                {/* Transition on Downbeat Toggle */}
                <div className="flex items-center justify-between bg-dark-800 rounded-retro p-3">
                  <div>
                    <span className="text-xs text-metal-silver font-mono">Transition on Downbeat</span>
                    <p className="text-xs text-metal-silver opacity-50 mt-0.5">Sync playlist transitions to downbeats</p>
                  </div>
                  <button
                    onClick={() => updateTransitionOnDownbeat(!transitionOnDownbeat)}
                    disabled={transitionOnDownbeatLoading}
                    className={`px-4 py-1 rounded-retro text-xs font-retro font-bold transition-all duration-200 ${
                      transitionOnDownbeat
                        ? 'bg-neon-cyan bg-opacity-20 text-neon-cyan border border-neon-cyan border-opacity-50'
                        : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30'
                    } ${transitionOnDownbeatLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {transitionOnDownbeatLoading ? '...' : transitionOnDownbeat ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Test Trigger Interval */}
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
                    onChange={(e) => {
                      const val = parseFloat(e.target.value)
                      setTestTriggerInterval(val)
                      updateServerConfig({ test_interval: val })
                    }}
                    className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                  />
                </div>
              </div>
            </div>

            {/* 🌐 COMMON RULES Section */}
            <div className="pl-4 border-l-2 border-neon-green border-opacity-30">
              <button
                onClick={() => toggleSection('common')}
                className="w-full flex items-center justify-between text-sm font-retro text-neon-green mb-3 hover:text-neon-cyan transition-colors"
              >
                <div className="flex items-center gap-2">
                  {collapsedSections.common ? <ChevronRightIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
                  <span>🌐 COMMON RULES (Always Active - First Priority)</span>
                </div>
                <span className="text-xs text-metal-silver">{commonRules.length} rules</span>
              </button>

              {!collapsedSections.common && (
                <div className="space-y-3">
                  <p className="text-xs text-metal-silver font-mono mb-2">
                    Common rules are checked first on every beat. If no common rule matches, carousel rules are checked.
                  </p>

                  {commonRules.length === 0 && (
                    <div className="text-center py-4 text-metal-silver text-sm font-mono opacity-70">
                      No common rules configured
                    </div>
                  )}

                  {commonRules.map((rule, index) => (
                    <div key={rule.id} className="bg-dark-800 rounded-retro border border-metal-silver border-opacity-20 p-4">
                      <TriggerEffectRule
                        rule={rule}
                        index={index}
                        totalRules={commonRules.length}
                        onDelete={() => deleteCommonRule(rule.id)}
                        onMoveUp={() => moveCommonRule(rule.id, 'up')}
                        onMoveDown={() => moveCommonRule(rule.id, 'down')}
                        onUpdateTriggerType={(type) => updateTriggerType(rule.id, type, true)}
                        onUpdateTriggerParam={(key, value) => updateTriggerParam(rule.id, key, value, true)}
                        onUpdateEffectClass={(cls) => updateEffectClass(rule.id, cls, true)}
                        onUpdateEffectParam={(key, value) => updateEffectParam(rule.id, key, value, true)}
                      />
                    </div>
                  ))}

                  <button
                    onClick={addCommonRule}
                    className="retro-button px-3 py-1 text-neon-green text-xs font-retro font-bold w-full"
                  >
                    <PlusIcon className="w-4 h-4 inline mr-1" />
                    ADD COMMON RULE
                  </button>
                </div>
              )}
            </div>

            {/* 📋 CAROUSEL RULE SETS Section */}
            <div className="pl-4 border-l-2 border-neon-purple border-opacity-30">
              <button
                onClick={() => toggleSection('carousel')}
                className="w-full flex items-center justify-between text-sm font-retro text-neon-purple mb-3 hover:text-neon-cyan transition-colors"
              >
                <div className="flex items-center gap-2">
                  {collapsedSections.carousel ? <ChevronRightIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
                  <span>📋 CAROUSEL RULE SETS (Fallback - Rotates)</span>
                </div>
                <span className="text-xs text-metal-silver">{carouselRuleSets.length} sets</span>
              </button>

              {!collapsedSections.carousel && (
                <div className="space-y-3">
                  <p className="text-xs text-metal-silver font-mono mb-2">
                    Carousel rule sets rotate every N beats. Only the current active set is checked (after common rules).
                  </p>

                  {/* Beat interval control */}
                  <div className="bg-dark-800 rounded-retro border border-neon-purple border-opacity-20 p-3">
                    <label className="block text-xs text-metal-silver font-mono mb-2">
                      🔄 Switch every {carouselBeatInterval} beats
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="32"
                      step="1"
                      value={carouselBeatInterval}
                      onChange={(e) => {
                        const val = parseInt(e.target.value)
                        setCarouselBeatInterval(val)
                        updateServerConfig({ carousel_beat_interval: val })
                      }}
                      className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                    />
                    <div className="text-xs text-metal-silver font-mono text-center mt-1">
                      1 beat → {carouselBeatInterval} beats → 32 beats
                    </div>
                  </div>

                  {carouselRuleSets.length === 0 && (
                    <div className="text-center py-4 text-metal-silver text-sm font-mono opacity-70">
                      No carousel rule sets configured
                    </div>
                  )}

                  {carouselRuleSets.map((ruleSet, setIndex) => (
                    <div key={ruleSet.id} className="bg-dark-800 rounded-retro border border-neon-purple border-opacity-30 p-4 space-y-3">
                      {/* Rule Set Header */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-1">
                          {editingSetId === ruleSet.id ? (
                            <>
                              <input
                                type="text"
                                value={editingSetName}
                                onChange={(e) => setEditingSetName(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') saveSetName(ruleSet.id)
                                  if (e.key === 'Escape') cancelEditingSetName()
                                }}
                                className="retro-input text-sm flex-1"
                                autoFocus
                              />
                              <button
                                onClick={() => saveSetName(ruleSet.id)}
                                className="p-1 rounded text-neon-green hover:text-neon-cyan"
                                title="Save"
                              >
                                <CheckIcon className="w-4 h-4" />
                              </button>
                              <button
                                onClick={cancelEditingSetName}
                                className="p-1 rounded text-metal-silver hover:text-neon-red"
                                title="Cancel"
                              >
                                <XMarkIcon className="w-4 h-4" />
                              </button>
                            </>
                          ) : (
                            <>
                              <span className="text-sm font-retro text-neon-purple">{ruleSet.name}</span>
                              <button
                                onClick={() => startEditingSetName(ruleSet.id, ruleSet.name)}
                                className="p-1 rounded text-metal-silver hover:text-neon-cyan"
                                title="Edit name"
                              >
                                <PencilIcon className="w-3 h-3" />
                              </button>
                            </>
                          )}
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="text-xs text-metal-silver font-mono">{ruleSet.rules.length} rules</span>
                          <button
                            onClick={() => moveCarouselRuleSet(ruleSet.id, 'up')}
                            disabled={setIndex === 0}
                            className={`p-1 rounded ${setIndex === 0 ? 'text-dark-600 cursor-not-allowed' : 'text-metal-silver hover:text-neon-cyan'}`}
                            title="Move up"
                          >
                            <ArrowUpIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => moveCarouselRuleSet(ruleSet.id, 'down')}
                            disabled={setIndex === carouselRuleSets.length - 1}
                            className={`p-1 rounded ${setIndex === carouselRuleSets.length - 1 ? 'text-dark-600 cursor-not-allowed' : 'text-metal-silver hover:text-neon-cyan'}`}
                            title="Move down"
                          >
                            <ArrowDownIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => deleteCarouselRuleSet(ruleSet.id)}
                            className="p-1 rounded text-metal-silver hover:text-neon-red"
                            title="Delete rule set"
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {/* Rules within this set */}
                      {ruleSet.rules.length === 0 ? (
                        <div className="text-center py-2 text-metal-silver text-xs font-mono opacity-70">
                          No rules in this set
                        </div>
                      ) : (
                        <div className="space-y-2 pl-3 border-l-2 border-neon-purple border-opacity-20">
                          {ruleSet.rules.map((rule, ruleIndex) => (
                            <div key={rule.id} className="bg-dark-700 rounded-retro border border-metal-silver border-opacity-10 p-3">
                              <TriggerEffectRule
                                rule={rule}
                                index={ruleIndex}
                                totalRules={ruleSet.rules.length}
                                onDelete={() => deleteCarouselRule(ruleSet.id, rule.id)}
                                onMoveUp={() => moveCarouselRule(ruleSet.id, rule.id, 'up')}
                                onMoveDown={() => moveCarouselRule(ruleSet.id, rule.id, 'down')}
                                onUpdateTriggerType={(type) => updateTriggerType(rule.id, type, false, ruleSet.id)}
                                onUpdateTriggerParam={(key, value) => updateTriggerParam(rule.id, key, value, false, ruleSet.id)}
                                onUpdateEffectClass={(cls) => updateEffectClass(rule.id, cls, false, ruleSet.id)}
                                onUpdateEffectParam={(key, value) => updateEffectParam(rule.id, key, value, false, ruleSet.id)}
                              />
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Add rule to this set */}
                      <button
                        onClick={() => addCarouselRule(ruleSet.id)}
                        className="retro-button px-3 py-1 text-neon-cyan text-xs font-retro font-bold w-full"
                      >
                        <PlusIcon className="w-4 h-4 inline mr-1" />
                        ADD RULE TO {ruleSet.name.toUpperCase()}
                      </button>
                    </div>
                  ))}

                  <button
                    onClick={addCarouselRuleSet}
                    className="retro-button px-3 py-1 text-neon-purple text-xs font-retro font-bold w-full"
                  >
                    <PlusIcon className="w-4 h-4 inline mr-1" />
                    ADD RULE SET
                  </button>
                </div>
              )}
            </div>

            {/* ✂️ CUT EFFECTS Section */}
            <div className="pl-4 border-l-2 border-neon-orange border-opacity-30">
              <button
                onClick={() => toggleSection('cutEffect')}
                className="w-full flex items-center justify-between text-sm font-retro text-neon-orange mb-3 hover:text-neon-cyan transition-colors"
              >
                <div className="flex items-center gap-2">
                  {collapsedSections.cutEffect ? <ChevronRightIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
                  <span>✂️ CUT EFFECTS (Energy Drop Detection)</span>
                </div>
                <span className="text-xs text-metal-silver">{cutEffect.enabled ? 'ON' : 'OFF'}</span>
              </button>

              {!collapsedSections.cutEffect && (
                <div className="space-y-3">
                  <p className="text-xs text-metal-silver font-mono mb-2">
                    Triggered when audio energy suddenly drops (e.g., breakdown, silence before drop).
                  </p>

                  {/* Enable/Disable Toggle */}
                  <div className="flex items-center justify-between bg-dark-800 rounded-retro p-3">
                    <span className="text-xs text-metal-silver font-mono">Enable Cut Effect</span>
                    <button
                      onClick={() => updateCutEffectEnabled(!cutEffect.enabled)}
                      className={`px-4 py-1 rounded-retro text-xs font-retro font-bold transition-all duration-200 ${
                        cutEffect.enabled
                          ? 'bg-neon-orange bg-opacity-20 text-neon-orange border border-neon-orange border-opacity-50'
                          : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30'
                      }`}
                    >
                      {cutEffect.enabled ? 'ENABLED' : 'DISABLED'}
                    </button>
                  </div>

                  {cutEffect.enabled && (
                    <EventEffectConfig
                      effectConfig={cutEffect}
                      onUpdateEffectClass={updateCutEffectClass}
                      onUpdateEffectParam={updateCutEffectParam}
                    />
                  )}
                </div>
              )}
            </div>

            {/* 💥 DROP EFFECTS Section */}
            <div className="pl-4 border-l-2 border-neon-red border-opacity-30">
              <button
                onClick={() => toggleSection('dropEffect')}
                className="w-full flex items-center justify-between text-sm font-retro text-neon-red mb-3 hover:text-neon-cyan transition-colors"
              >
                <div className="flex items-center gap-2">
                  {collapsedSections.dropEffect ? <ChevronRightIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
                  <span>💥 DROP EFFECTS (Bass Drop Detection)</span>
                </div>
                <span className="text-xs text-metal-silver">{dropEffect.enabled ? 'ON' : 'OFF'}</span>
              </button>

              {!collapsedSections.dropEffect && (
                <div className="space-y-3">
                  <p className="text-xs text-metal-silver font-mono mb-2">
                    Triggered when bass energy suddenly increases after a buildup (e.g., the drop).
                  </p>

                  {/* Enable/Disable Toggle */}
                  <div className="flex items-center justify-between bg-dark-800 rounded-retro p-3">
                    <span className="text-xs text-metal-silver font-mono">Enable Drop Effect</span>
                    <button
                      onClick={() => updateDropEffectEnabled(!dropEffect.enabled)}
                      className={`px-4 py-1 rounded-retro text-xs font-retro font-bold transition-all duration-200 ${
                        dropEffect.enabled
                          ? 'bg-neon-red bg-opacity-20 text-neon-red border border-neon-red border-opacity-50'
                          : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30'
                      }`}
                    >
                      {dropEffect.enabled ? 'ENABLED' : 'DISABLED'}
                    </button>
                  </div>

                  {dropEffect.enabled && (
                    <EventEffectConfig
                      effectConfig={dropEffect}
                      onUpdateEffectClass={updateDropEffectClass}
                      onUpdateEffectParam={updateDropEffectParam}
                    />
                  )}
                </div>
              )}
            </div>

            {/* ✨ SPARKLE EFFECTS Section */}
            <div className="pl-4 border-l-2 border-neon-yellow border-opacity-30">
              <button
                onClick={() => toggleSection('sparkleEffect')}
                className="w-full flex items-center justify-between text-sm font-retro text-neon-yellow mb-3 hover:text-neon-cyan transition-colors"
              >
                <div className="flex items-center gap-2">
                  {collapsedSections.sparkleEffect ? <ChevronRightIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
                  <span>✨ SPARKLE EFFECTS (Buildup Detection)</span>
                </div>
                <span className="text-xs text-metal-silver">{sparkleEffect.enabled ? 'ON' : 'OFF'}</span>
              </button>

              {!collapsedSections.sparkleEffect && (
                <div className="space-y-4">
                  <p className="text-xs text-metal-silver font-mono mb-2">
                    Creates sparkle/glitter effect during buildups. Parameters are derived from buildup intensity using configurable ranges and curves.
                  </p>

                  {/* Enable/Disable Toggle */}
                  <div className="flex items-center justify-between bg-dark-800 rounded-retro p-3">
                    <span className="text-xs text-metal-silver font-mono">Enable Sparkle Effect</span>
                    <button
                      onClick={() => updateSparkleEffectEnabled(!sparkleEffect.enabled)}
                      className={`px-4 py-1 rounded-retro text-xs font-retro font-bold transition-all duration-200 ${
                        sparkleEffect.enabled
                          ? 'bg-neon-yellow bg-opacity-20 text-neon-yellow border border-neon-yellow border-opacity-50'
                          : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30'
                      }`}
                    >
                      {sparkleEffect.enabled ? 'ENABLED' : 'DISABLED'}
                    </button>
                  </div>

                  {sparkleEffect.enabled && (
                    <div className="space-y-4 bg-dark-800 rounded-retro p-4">
                      {/* Random Colors Toggle */}
                      <div className="flex items-center justify-between py-2 border-b border-dark-600">
                        <div>
                          <span className="text-xs text-metal-silver font-mono">Random Colors</span>
                          <p className="text-xs text-metal-silver opacity-50 mt-0.5">Use random hue colors instead of white</p>
                        </div>
                        <button
                          onClick={() => updateSparkleEffectRandomColors(!sparkleEffect.random_colors)}
                          className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
                            sparkleEffect.random_colors
                              ? 'bg-neon-cyan bg-opacity-20 text-neon-cyan border border-neon-cyan border-opacity-50'
                              : 'bg-dark-700 text-metal-silver border border-metal-silver border-opacity-30'
                          }`}
                        >
                          {sparkleEffect.random_colors ? 'ON' : 'OFF'}
                        </button>
                      </div>

                      {/* LED Density Configuration */}
                      <div className="space-y-3">
                        <h5 className="text-xs font-retro text-neon-cyan">LED DENSITY</h5>
                        <p className="text-xs text-metal-silver font-mono opacity-70">
                          Fraction of LEDs that sparkle per burst (low intensity → min, high intensity → max)
                        </p>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <label className="block text-xs text-metal-silver font-mono">
                              Min Density ({((sparkleEffect.density?.min || 0.01) * 100).toFixed(0)}%)
                            </label>
                            <input
                              type="range"
                              min="0"
                              max="0.5"
                              step="0.01"
                              value={sparkleEffect.density?.min || 0.01}
                              onChange={(e) => updateSparkleEffectDensity('min', parseFloat(e.target.value))}
                              className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                            />
                          </div>
                          <div className="space-y-1">
                            <label className="block text-xs text-metal-silver font-mono">
                              Max Density ({((sparkleEffect.density?.max || 0.15) * 100).toFixed(0)}%)
                            </label>
                            <input
                              type="range"
                              min="0.01"
                              max="1.0"
                              step="0.01"
                              value={sparkleEffect.density?.max || 0.15}
                              onChange={(e) => updateSparkleEffectDensity('max', parseFloat(e.target.value))}
                              className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                            />
                          </div>
                        </div>

                        <div className="space-y-1">
                          <label className="block text-xs text-metal-silver font-mono">Response Curve</label>
                          <select
                            value={sparkleEffect.density?.curve || 'linear'}
                            onChange={(e) => updateSparkleEffectDensity('curve', e.target.value)}
                            className="retro-input w-full text-sm"
                          >
                            {SPARKLE_CURVE_OPTIONS.map(curve => (
                              <option key={curve} value={curve}>{curve}</option>
                            ))}
                          </select>
                        </div>
                      </div>

                      {/* Sparkle Interval Configuration */}
                      <div className="space-y-3">
                        <h5 className="text-xs font-retro text-neon-cyan">SPARKLE INTERVAL</h5>
                        <p className="text-xs text-metal-silver font-mono opacity-70">
                          Time between sparkle bursts in ms (inverse: high intensity → faster sparkles)
                        </p>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <label className="block text-xs text-metal-silver font-mono">
                              Min Interval ({sparkleEffect.interval_ms?.min || 30}ms)
                            </label>
                            <input
                              type="range"
                              min="10"
                              max="100"
                              step="5"
                              value={sparkleEffect.interval_ms?.min || 30}
                              onChange={(e) => updateSparkleEffectInterval('min', parseFloat(e.target.value))}
                              className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                            />
                          </div>
                          <div className="space-y-1">
                            <label className="block text-xs text-metal-silver font-mono">
                              Max Interval ({sparkleEffect.interval_ms?.max || 300}ms)
                            </label>
                            <input
                              type="range"
                              min="100"
                              max="1000"
                              step="10"
                              value={sparkleEffect.interval_ms?.max || 300}
                              onChange={(e) => updateSparkleEffectInterval('max', parseFloat(e.target.value))}
                              className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                            />
                          </div>
                        </div>

                        <div className="space-y-1">
                          <label className="block text-xs text-metal-silver font-mono">Response Curve</label>
                          <select
                            value={sparkleEffect.interval_ms?.curve || 'inverse'}
                            onChange={(e) => updateSparkleEffectInterval('curve', e.target.value)}
                            className="retro-input w-full text-sm"
                          >
                            {SPARKLE_CURVE_OPTIONS.map(curve => (
                              <option key={curve} value={curve}>{curve}</option>
                            ))}
                          </select>
                        </div>
                      </div>

                      {/* Fade Multiplier */}
                      <div className="space-y-3">
                        <h5 className="text-xs font-retro text-neon-cyan">FADE DURATION</h5>
                        <p className="text-xs text-metal-silver font-mono opacity-70">
                          Fade duration = interval × multiplier
                        </p>

                        <div className="space-y-1">
                          <label className="block text-xs text-metal-silver font-mono">
                            Fade Multiplier ({(sparkleEffect.fade_multiplier || 2.0).toFixed(1)}x)
                          </label>
                          <input
                            type="range"
                            min="0.5"
                            max="5.0"
                            step="0.1"
                            value={sparkleEffect.fade_multiplier || 2.0}
                            onChange={(e) => updateSparkleEffectFadeMultiplier(parseFloat(e.target.value))}
                            className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer slider-track"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Info Panel */}
      <div className="mt-4 p-3 bg-dark-800 rounded-retro border border-neon-purple border-opacity-20">
        <p className="text-xs text-metal-silver font-mono leading-relaxed">
          <span className="text-neon-purple">INFO:</span> Common rules are always checked first. If no common rule matches,
          the current carousel rule set is checked. Carousel sets rotate every N beats for variety.
          Cut/Drop effects trigger on audio energy transitions.
        </p>
      </div>
    </div>
  )
}

const TriggerEffectRule = ({
  rule,
  index,
  totalRules,
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
        <span className="text-sm font-retro text-neon-cyan">Rule {index + 1}</span>

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

// Event Effect Configuration Component (for Cut/Drop effects)
const EventEffectConfig = ({ effectConfig, onUpdateEffectClass, onUpdateEffectParam }) => {
  const effectType = EVENT_EFFECT_TYPES[effectConfig.effect_class] || EVENT_EFFECT_TYPES.none

  return (
    <div className="space-y-4 bg-dark-800 rounded-retro p-4">
      {/* Effect Type Selector */}
      <div>
        <h5 className="text-xs font-retro text-neon-yellow mb-2">EFFECT TYPE</h5>
        <div className="flex items-center gap-2">
          <span className="text-lg">{effectType.icon}</span>
          <select
            value={effectConfig.effect_class}
            onChange={(e) => onUpdateEffectClass(e.target.value)}
            className="retro-input flex-1 text-sm"
          >
            {Object.entries(EVENT_EFFECT_TYPES).map(([key, config]) => (
              <option key={key} value={key}>
                {config.name}
              </option>
            ))}
          </select>
        </div>
        <p className="text-xs text-metal-silver font-mono mt-1">{effectType.description}</p>
      </div>

      {/* Effect Parameters */}
      {effectType.params.length > 0 && (
        <div className="space-y-3">
          <h5 className="text-xs font-retro text-neon-orange mb-2">PARAMETERS</h5>
          {effectType.params.map(param => (
            shouldShowParam(param, effectConfig.params) && (
              <ParameterInput
                key={param.key}
                param={param}
                value={effectConfig.params[param.key]}
                onChange={(value) => onUpdateEffectParam(param.key, value)}
              />
            )
          ))}
        </div>
      )}
    </div>
  )
}

export default AudioReactivePanel
