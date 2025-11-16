// Audio Reactive Trigger and Effect Configuration Registry

// Template name to file path mapping
export const TEMPLATE_OPTIONS = {
  'Ring': 'templates/ring_800x480_leds.npy',
  'Wide Ring': 'templates/wide_ring_800x480_leds.npy',
  'Heart': 'templates/heart_800x480_leds.npy',
  'Star': 'templates/star7_800x480_leds.npy'
}

// Trigger Type Registry
export const TRIGGER_TYPES = {
  beat: {
    name: 'Beat Detection',
    icon: '🎵',
    description: 'Trigger on detected audio beats',
    params: [
      {
        key: 'confidence_min',
        label: 'Min Confidence',
        type: 'slider',
        min: 0,
        max: 1,
        step: 0.01,
        default: null,
        optional: true,
        description: 'Minimum beat confidence (0-100%)'
      },
      {
        key: 'intensity_min',
        label: 'Min Intensity',
        type: 'slider',
        min: 0,
        max: 1,
        step: 0.01,
        default: null,
        optional: true,
        description: 'Minimum beat intensity'
      },
      {
        key: 'bpm_min',
        label: 'Min BPM',
        type: 'number',
        min: 0,
        max: 300,
        default: null,
        optional: true,
        description: 'Minimum BPM threshold'
      },
      {
        key: 'bpm_max',
        label: 'Max BPM',
        type: 'number',
        min: 0,
        max: 300,
        default: null,
        optional: true,
        description: 'Maximum BPM threshold'
      }
    ]
  },
  test: {
    name: 'Test (Periodic)',
    icon: '🧪',
    description: 'Trigger periodically for testing (uses global interval)',
    params: [] // Uses global test_trigger_interval
  }
}

// Effect Type Registry
export const EFFECT_TYPES = {
  BeatBrightnessEffect: {
    name: 'Beat Brightness Boost',
    icon: '💡',
    description: 'Sine wave brightness boost synchronized to beat',
    params: [
      {
        key: 'boost_intensity',
        label: 'Boost Intensity',
        type: 'slider',
        min: 0,
        max: 5,
        step: 0.1,
        default: 4.0,
        description: 'Brightness boost multiplier (0-5x)'
      },
      {
        key: 'duration_fraction',
        label: 'Duration Fraction',
        type: 'slider',
        min: 0.1,
        max: 1.0,
        step: 0.05,
        default: 0.4,
        description: 'Fraction of beat interval for boost (10-100%)'
      }
    ],
    providedByTrigger: ['bpm', 'beat_intensity', 'beat_confidence'] // Auto-provided by beat trigger
  },
  TemplateEffect: {
    name: 'Template Animation',
    icon: '🎨',
    description: 'Apply pre-rendered LED pattern template',
    params: [
      {
        key: 'template_path',
        label: 'Template',
        type: 'template_select',
        options: TEMPLATE_OPTIONS,
        default: 'templates/ring_800x480_leds.npy',
        description: 'LED pattern template to apply'
      },
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 1.0,
        description: 'Effect duration in seconds'
      },
      {
        key: 'blend_mode',
        label: 'Blend Mode',
        type: 'select',
        options: ['alpha', 'add', 'multiply', 'replace', 'boost', 'addboost'],
        default: 'addboost',
        description: 'How to blend template with base LED values'
      },
      {
        key: 'intensity_multiplier',
        label: 'Intensity Multiplier',
        type: 'slider',
        min: 0,
        max: 5,
        step: 0.1,
        default: 2.0,
        description: 'Beat intensity multiplier for effect strength (multiplied by beat intensity)'
      },
      {
        key: 'add_multiplier_factor',
        label: 'Add Multiplier',
        type: 'slider',
        min: 0,
        max: 2,
        step: 0.1,
        default: 0.4,
        description: 'Beat intensity multiplier for additive component (for addboost mode)',
        showIf: { blend_mode: 'addboost' }
      }
    ]
  }
}

// Helper function to get default parameter values for a trigger type
export function getDefaultTriggerParams(triggerType) {
  const trigger = TRIGGER_TYPES[triggerType]
  if (!trigger) return {}

  const params = {}
  trigger.params.forEach(param => {
    params[param.key] = param.default
  })
  return params
}

// Helper function to get default parameter values for an effect type
export function getDefaultEffectParams(effectType) {
  const effect = EFFECT_TYPES[effectType]
  if (!effect) return {}

  const params = {}
  effect.params.forEach(param => {
    params[param.key] = param.default
  })
  return params
}

// Helper function to check if a parameter should be shown
export function shouldShowParam(param, currentParams) {
  if (!param.showIf) return true

  // Check all conditions in showIf
  for (const [key, value] of Object.entries(param.showIf)) {
    if (currentParams[key] !== value) {
      return false
    }
  }
  return true
}

// Helper function to get friendly template name from path
export function getTemplateName(templatePath) {
  for (const [name, path] of Object.entries(TEMPLATE_OPTIONS)) {
    if (path === templatePath) return name
  }
  return templatePath // Fallback to path if not found
}

// Helper function to get template path from friendly name
export function getTemplatePath(templateName) {
  return TEMPLATE_OPTIONS[templateName] || templateName
}
