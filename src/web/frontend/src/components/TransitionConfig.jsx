import React, { useState, useEffect } from 'react'
import {
  Cog6ToothIcon,
  XMarkIcon,
  ChevronDownIcon
} from '@heroicons/react/24/outline'

const TransitionConfig = ({ item, onUpdate, onClose }) => {
  const [transitionIn, setTransitionIn] = useState({
    type: 'none',
    parameters: {}
  })
  const [transitionOut, setTransitionOut] = useState({
    type: 'none',
    parameters: {}
  })
  const [availableTransitions, setAvailableTransitions] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [errors, setErrors] = useState({})

  // Load available transition types on mount
  useEffect(() => {
    const loadTransitionTypes = async () => {
      try {
        const response = await fetch('/api/transitions')
        if (response.ok) {
          const data = await response.json()
          setAvailableTransitions(data.types)
        }
      } catch (error) {
        console.error('Failed to load transition types:', error)
      }
    }
    loadTransitionTypes()
  }, [])

  // Initialize with current item transitions
  useEffect(() => {
    if (item) {
      setTransitionIn(item.transition_in || { type: 'none', parameters: {} })
      setTransitionOut(item.transition_out || { type: 'none', parameters: {} })
    }
  }, [item])

  const getTransitionSchema = (type) => {
    const transition = availableTransitions.find(t => t.type === type)
    return transition ? transition.parameters : {}
  }

  const handleTransitionTypeChange = (direction, newType) => {
    const schema = getTransitionSchema(newType)
    const defaultParams = {}

    // Set default values based on schema
    Object.entries(schema).forEach(([key, paramSchema]) => {
      defaultParams[key] = paramSchema.default
    })

    if (direction === 'in') {
      setTransitionIn({ type: newType, parameters: defaultParams })
    } else {
      setTransitionOut({ type: newType, parameters: defaultParams })
    }

    // Clear any errors for this direction
    setErrors(prev => {
      const newErrors = { ...prev }
      Object.keys(newErrors).forEach(key => {
        if (key.startsWith(`transition_${direction}.`)) {
          delete newErrors[key]
        }
      })
      return newErrors
    })
  }

  const handleParameterChange = (direction, paramName, value) => {
    const numericValue = paramName === 'duration' ? parseFloat(value) : value

    if (direction === 'in') {
      setTransitionIn(prev => ({
        ...prev,
        parameters: { ...prev.parameters, [paramName]: numericValue }
      }))
    } else {
      setTransitionOut(prev => ({
        ...prev,
        parameters: { ...prev.parameters, [paramName]: numericValue }
      }))
    }

    // Clear error for this parameter
    const errorKey = `transition_${direction}.parameters.${paramName}`
    if (errors[errorKey]) {
      setErrors(prev => {
        const newErrors = { ...prev }
        delete newErrors[errorKey]
        return newErrors
      })
    }
  }

  const handleSave = async () => {
    setIsLoading(true)
    setErrors({})

    try {
      const response = await fetch(`/api/playlist/${item.id}/transitions`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          transition_in: transitionIn,
          transition_out: transitionOut
        })
      })

      if (response.ok) {
        const result = await response.json()
        onUpdate && onUpdate(result)
        onClose && onClose()
      } else {
        const errorData = await response.json()
        if (errorData.detail && errorData.detail.errors) {
          setErrors(errorData.detail.errors)
        } else {
          setErrors({ general: 'Failed to update transitions' })
        }
      }
    } catch (error) {
      console.error('Failed to update transitions:', error)
      setErrors({ general: 'Network error occurred' })
    } finally {
      setIsLoading(false)
    }
  }

  const renderParameterInput = (direction, paramName, paramSchema) => {
    const currentValue = direction === 'in'
      ? transitionIn.parameters[paramName]
      : transitionOut.parameters[paramName]
    const errorKey = `transition_${direction}.parameters.${paramName}`
    const error = errors[errorKey]

    return (
      <div key={paramName} className="space-y-1">
        <label className="block text-xs font-mono text-metal-silver uppercase">
          {paramName}
        </label>
        <input
          type="number"
          value={currentValue || ''}
          onChange={(e) => handleParameterChange(direction, paramName, e.target.value)}
          min={paramSchema.min}
          max={paramSchema.max}
          step={paramName === 'duration' ? 0.1 : 1}
          className={`w-full px-3 py-2 bg-metal-dark border rounded-retro font-mono text-sm
            ${error ? 'border-neon-orange' : 'border-metal-silver'}
            text-black focus:border-neon-cyan focus:outline-none`}
          placeholder={paramSchema.default}
        />
        {error && (
          <p className="text-xs text-neon-orange font-mono">{error}</p>
        )}
        <p className="text-xs text-metal-silver font-mono">{paramSchema.description}</p>
      </div>
    )
  }

  const renderTransitionConfig = (direction, transition, onTypeChange) => {
    const schema = getTransitionSchema(transition.type)
    const directionLabel = direction === 'in' ? 'TRANSITION IN' : 'TRANSITION OUT'
    const errorKey = `transition_${direction}.type`
    const typeError = errors[errorKey]

    return (
      <div className="space-y-4">
        <h4 className="text-sm font-retro text-neon-cyan uppercase">{directionLabel}</h4>

        {/* Transition Type Selector */}
        <div className="space-y-1">
          <label className="block text-xs font-mono text-metal-silver uppercase">
            Type
          </label>
          <div className="relative">
            <select
              value={transition.type}
              onChange={(e) => onTypeChange(direction, e.target.value)}
              className={`w-full px-3 py-2 bg-metal-dark border rounded-retro font-mono text-sm appearance-none cursor-pointer
                ${typeError ? 'border-neon-orange' : 'border-metal-silver'}
                text-black focus:border-neon-cyan focus:outline-none`}
            >
              {availableTransitions.map(trans => (
                <option key={trans.type} value={trans.type}>
                  {trans.name}
                </option>
              ))}
            </select>
            <ChevronDownIcon className="absolute right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-metal-silver pointer-events-none" />
          </div>
          {typeError && (
            <p className="text-xs text-neon-orange font-mono">{typeError}</p>
          )}
        </div>

        {/* Parameters */}
        {Object.keys(schema).length > 0 && (
          <div className="space-y-3">
            <h5 className="text-xs font-mono text-metal-silver uppercase">Parameters</h5>
            {Object.entries(schema).map(([paramName, paramSchema]) =>
              renderParameterInput(direction, paramName, paramSchema)
            )}
          </div>
        )}
      </div>
    )
  }

  if (!item) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="retro-container max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-retro text-neon-cyan">TRANSITION CONFIG</h3>
            <p className="text-sm text-metal-silver font-mono">{item.name}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-metal-silver hover:text-neon-orange"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* General Error */}
        {errors.general && (
          <div className="mb-4 p-3 bg-neon-orange bg-opacity-10 border border-neon-orange rounded-retro">
            <p className="text-sm text-neon-orange font-mono">{errors.general}</p>
          </div>
        )}

        {/* Transition Configurations */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {renderTransitionConfig('in', transitionIn, handleTransitionTypeChange)}
          {renderTransitionConfig('out', transitionOut, handleTransitionTypeChange)}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-metal-silver text-metal-silver hover:border-neon-cyan hover:text-neon-cyan rounded-retro font-mono text-sm transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isLoading}
            className="px-4 py-2 bg-neon-cyan text-metal-dark hover:bg-neon-pink rounded-retro font-mono text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default TransitionConfig
