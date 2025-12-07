// Audio Reactive Trigger and Effect Configuration Registry

// Template name to file path mapping
export const TEMPLATE_OPTIONS = {
  'Ring': 'templates/ring_800x480_leds.npy',
  'Wide Ring': 'templates/wide_ring_800x480_leds.npy',
  'Heart': 'templates/heart_800x480_leds.npy',
  'Star': 'templates/star7_800x480_leds.npy'
}

// Interpolation curve options
export const CURVE_OPTIONS = ['linear', 'ease-in', 'ease-out', 'ease-in-out']

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
      },
      {
        key: 'color_thieving',
        label: 'Color Thieving',
        type: 'checkbox',
        default: false,
        description: 'Extract color from input LEDs and apply to template (for add/addboost modes)'
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

// ========================================================================================
// Event Effect Types (for Cut/Drop events)
// ========================================================================================

// Event Effect Type Registry - effects that can be triggered by cut/drop audio events
export const EVENT_EFFECT_TYPES = {
  none: {
    name: 'None (Disabled)',
    icon: '🚫',
    description: 'No effect on this event',
    params: []
  },
  FadeInEffect: {
    name: 'Fade In (from Black)',
    icon: '🌅',
    description: 'Fade from black to content over duration',
    params: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 2.0,
        description: 'Fade duration in seconds'
      },
      {
        key: 'curve',
        label: 'Curve',
        type: 'select',
        options: CURVE_OPTIONS,
        default: 'ease-in',
        description: 'Interpolation curve'
      },
      {
        key: 'min_brightness',
        label: 'Min Brightness',
        type: 'slider',
        min: 0,
        max: 0.5,
        step: 0.01,
        default: 0.0,
        description: 'Starting brightness level'
      }
    ]
  },
  FadeOutEffect: {
    name: 'Fade Out (to Black)',
    icon: '🌑',
    description: 'Fade from content to black over duration',
    params: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 2.0,
        description: 'Fade duration in seconds'
      },
      {
        key: 'curve',
        label: 'Curve',
        type: 'select',
        options: CURVE_OPTIONS,
        default: 'ease-out',
        description: 'Interpolation curve'
      },
      {
        key: 'min_brightness',
        label: 'Min Brightness',
        type: 'slider',
        min: 0,
        max: 0.5,
        step: 0.01,
        default: 0.0,
        description: 'Ending brightness level'
      }
    ]
  },
  InverseFadeIn: {
    name: 'Inverse Fade In (from White)',
    icon: '💫',
    description: 'Flash to white then fade back to content',
    params: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 2.0,
        description: 'Fade duration in seconds'
      },
      {
        key: 'curve',
        label: 'Curve',
        type: 'select',
        options: CURVE_OPTIONS,
        default: 'ease-out',
        description: 'Interpolation curve'
      }
    ]
  },
  InverseFadeOut: {
    name: 'Inverse Fade Out (to White)',
    icon: '☀️',
    description: 'Fade from content to white over duration',
    params: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 2.0,
        description: 'Fade duration in seconds'
      },
      {
        key: 'curve',
        label: 'Curve',
        type: 'select',
        options: CURVE_OPTIONS,
        default: 'ease-in',
        description: 'Interpolation curve'
      }
    ]
  },
  RandomInEffect: {
    name: 'Random Reveal',
    icon: '✨',
    description: 'LEDs light up in random order from black',
    params: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 2.0,
        description: 'Transition duration in seconds'
      },
      {
        key: 'leds_per_frame',
        label: 'LEDs per Frame',
        type: 'number',
        min: 1,
        max: 100,
        step: 1,
        default: 10,
        description: 'Number of LEDs to light per frame'
      },
      {
        key: 'fade_tail',
        label: 'Fade Tail',
        type: 'checkbox',
        default: true,
        description: 'Apply fade effect to recently lit LEDs'
      }
    ]
  },
  RandomOutEffect: {
    name: 'Random Blank',
    icon: '🌌',
    description: 'LEDs blank out in random order to black',
    params: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        min: 0.1,
        max: 10,
        step: 0.1,
        default: 2.0,
        description: 'Transition duration in seconds'
      },
      {
        key: 'leds_per_frame',
        label: 'LEDs per Frame',
        type: 'number',
        min: 1,
        max: 100,
        step: 1,
        default: 10,
        description: 'Number of LEDs to blank per frame'
      },
      {
        key: 'fade_tail',
        label: 'Fade Tail',
        type: 'checkbox',
        default: true,
        description: 'Apply fade effect to recently blanked LEDs'
      }
    ]
  },
  TemplateEffect: {
    name: 'Template Animation',
    icon: '🎨',
    description: 'Play a pre-rendered LED pattern template',
    params: [
      {
        key: 'template_path',
        label: 'Template',
        type: 'template_select',
        options: TEMPLATE_OPTIONS,
        default: 'templates/ring_800x480_leds.npy',
        description: 'LED pattern template to play'
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
        key: 'intensity',
        label: 'Intensity',
        type: 'slider',
        min: 0,
        max: 5,
        step: 0.1,
        default: 2.0,
        description: 'Effect intensity/opacity'
      },
      {
        key: 'add_multiplier',
        label: 'Add Multiplier',
        type: 'slider',
        min: 0,
        max: 2,
        step: 0.1,
        default: 0.4,
        description: 'Additive component strength (for addboost mode)',
        showIf: { blend_mode: 'addboost' }
      },
      {
        key: 'color_thieving',
        label: 'Color Thieving',
        type: 'checkbox',
        default: false,
        description: 'Extract color from input LEDs'
      }
    ]
  }
}

// Helper function to get default event effect params
export function getDefaultEventEffectParams(effectType) {
  const effect = EVENT_EFFECT_TYPES[effectType]
  if (!effect) return {}

  const params = {}
  effect.params.forEach(param => {
    params[param.key] = param.default
  })
  return params
}

// Default cut effect configuration
export const DEFAULT_CUT_EFFECT = {
  enabled: true,
  effect_class: 'FadeInEffect',
  params: {
    duration: 2.0,
    curve: 'ease-in',
    min_brightness: 0.0
  }
}

// Default drop effect configuration
export const DEFAULT_DROP_EFFECT = {
  enabled: true,
  effect_class: 'InverseFadeIn',
  params: {
    duration: 2.0,
    curve: 'ease-out'
  }
}

// ========================================================================================
// Sparkle Effect Configuration (triggered by buildup intensity)
// ========================================================================================

// Response curve options for sparkle parameter derivation
export const SPARKLE_CURVE_OPTIONS = ['linear', 'ease-in', 'ease-out', 'inverse']

// Sparkle effect configuration registry
// Maps buildup intensity (0-10) to sparkle parameters using configurable ranges and curves
export const SPARKLE_EFFECT_CONFIG = {
  // Density: fraction of LEDs that sparkle per burst
  // At intensity 0 -> min, at intensity 10 -> max
  density: {
    key: 'density',
    label: 'LED Density',
    description: 'Fraction of LEDs that sparkle per burst',
    min: { value: 0.01, label: 'Min Density', min: 0, max: 0.5, step: 0.01 },
    max: { value: 0.15, label: 'Max Density', min: 0.01, max: 1.0, step: 0.01 },
    curve: 'linear'  // linear, ease-in, ease-out
  },
  // Interval: milliseconds between sparkle bursts
  // At intensity 0 -> max (slow), at intensity 10 -> min (fast)
  interval_ms: {
    key: 'interval_ms',
    label: 'Sparkle Interval',
    description: 'Time between sparkle bursts (inverted: high intensity = faster)',
    min: { value: 30, label: 'Min Interval', min: 10, max: 100, step: 5 },
    max: { value: 300, label: 'Max Interval', min: 100, max: 1000, step: 10 },
    curve: 'inverse'  // inverse: high intensity = low value
  },
  // Fade: multiplier applied to interval to get fade duration
  fade_multiplier: {
    key: 'fade_multiplier',
    label: 'Fade Multiplier',
    description: 'Fade duration = interval × multiplier',
    value: 2.0,
    min: 0.5,
    max: 5.0,
    step: 0.1
  }
}

// Default sparkle effect configuration
export const DEFAULT_SPARKLE_EFFECT = {
  enabled: true,
  random_colors: false,
  density: {
    min: 0.01,
    max: 0.15,
    curve: 'linear'
  },
  interval_ms: {
    min: 30,
    max: 300,
    curve: 'inverse'
  },
  fade_multiplier: 2.0
}

// Helper function to get default sparkle effect config
export function getDefaultSparkleEffectConfig() {
  return { ...DEFAULT_SPARKLE_EFFECT }
}

// Helper function to calculate sparkle parameter from buildup intensity
// intensity: buildup intensity value (0 to ~10)
// paramConfig: { min, max, curve } configuration object
// Returns: calculated parameter value
export function calculateSparkleParam(intensity, paramConfig) {
  const { min, max, curve } = paramConfig

  // Clamp intensity to 0-10 range for calculation
  const clampedIntensity = Math.max(0, Math.min(10, intensity))
  // Normalize to 0-1
  const t = clampedIntensity / 10.0

  let curvedT
  switch (curve) {
    case 'ease-in':
      // Quadratic ease-in: slow start, fast end
      curvedT = t * t
      break
    case 'ease-out':
      // Quadratic ease-out: fast start, slow end
      curvedT = 1 - (1 - t) * (1 - t)
      break
    case 'inverse':
      // Inverse: high intensity = low value
      curvedT = 1 - t
      break
    case 'linear':
    default:
      curvedT = t
      break
  }

  // Interpolate between min and max
  return min + curvedT * (max - min)
}
