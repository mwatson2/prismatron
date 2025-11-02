import React from 'react'
import {
  ClockIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  XCircleIcon,
  StopIcon
} from '@heroicons/react/24/outline'

const ConversionProgress = ({ conversions, onCancel, onRemove }) => {
  if (!conversions || conversions.length === 0) {
    return null
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'queued':
        return <ClockIcon className="w-5 h-5 text-neon-cyan" />
      case 'processing':
        return <div className="w-5 h-5 border-2 border-neon-cyan border-t-transparent rounded-full animate-spin" />
      case 'validating':
        return <div className="w-5 h-5 border-2 border-neon-purple border-t-transparent rounded-full animate-spin" />
      case 'completed':
        return <CheckCircleIcon className="w-5 h-5 text-neon-green" />
      case 'failed':
        return <ExclamationCircleIcon className="w-5 h-5 text-neon-orange" />
      case 'cancelled':
        return <XCircleIcon className="w-5 h-5 text-metal-silver" />
      default:
        return <ClockIcon className="w-5 h-5 text-metal-silver" />
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'queued':
        return 'QUEUED'
      case 'processing':
        return 'CONVERTING'
      case 'validating':
        return 'VALIDATING'
      case 'completed':
        return 'COMPLETED'
      case 'failed':
        return 'FAILED'
      case 'cancelled':
        return 'CANCELLED'
      default:
        return 'UNKNOWN'
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'queued':
        return 'text-neon-cyan'
      case 'processing':
        return 'text-neon-cyan'
      case 'validating':
        return 'text-neon-purple'
      case 'completed':
        return 'text-neon-green'
      case 'failed':
        return 'text-neon-orange'
      case 'cancelled':
        return 'text-metal-silver'
      default:
        return 'text-metal-silver'
    }
  }

  const formatTime = (isoString) => {
    if (!isoString) return null
    const date = new Date(isoString)
    return date.toLocaleTimeString()
  }

  const formatDuration = (startTime, endTime) => {
    if (!startTime) return null
    const start = new Date(startTime)
    const end = endTime ? new Date(endTime) : new Date()
    const duration = Math.round((end - start) / 1000)

    if (duration < 60) {
      return `${duration}s`
    }
    const minutes = Math.floor(duration / 60)
    const seconds = duration % 60
    return `${minutes}m ${seconds}s`
  }

  return (
    <div className="retro-container">
      <h3 className="text-lg font-retro text-neon-purple mb-4">VIDEO CONVERSIONS</h3>

      <div className="space-y-3">
        {conversions.map((conversion) => (
          <div
            key={conversion.id}
            className={`border rounded-retro p-4 ${
              conversion.status === 'completed'
                ? 'border-neon-green border-opacity-30 bg-neon-green bg-opacity-5'
                : conversion.status === 'failed'
                ? 'border-neon-orange border-opacity-30 bg-neon-orange bg-opacity-5'
                : conversion.status === 'cancelled'
                ? 'border-metal-silver border-opacity-30 bg-metal-silver bg-opacity-5'
                : 'border-neon-cyan border-opacity-30 bg-neon-cyan bg-opacity-5'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                {getStatusIcon(conversion.status)}
                <div>
                  <p className="text-sm font-medium text-neon-cyan truncate">
                    {conversion.original_name}
                  </p>
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className={`uppercase ${getStatusColor(conversion.status)}`}>
                      {getStatusText(conversion.status)}
                    </span>
                    {conversion.started_at && (
                      <span className="text-metal-silver">
                        {formatTime(conversion.started_at)}
                      </span>
                    )}
                    {conversion.started_at && (
                      <span className="text-metal-silver">
                        {formatDuration(conversion.started_at, conversion.completed_at)}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {(conversion.status === 'queued' || conversion.status === 'processing' || conversion.status === 'validating') && (
                  <button
                    onClick={() => onCancel && onCancel(conversion.id)}
                    className="p-1 text-metal-silver hover:text-neon-orange transition-colors"
                    title="Cancel conversion"
                  >
                    <StopIcon className="w-4 h-4" />
                  </button>
                )}

                {(conversion.status === 'completed' || conversion.status === 'failed' || conversion.status === 'cancelled') && (
                  <button
                    onClick={() => onRemove && onRemove(conversion.id)}
                    className="p-1 text-metal-silver hover:text-neon-orange transition-colors"
                    title="Remove from list"
                  >
                    <XCircleIcon className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            {(conversion.status === 'processing' || conversion.status === 'validating') && (
              <div className="space-y-2">
                <div className="w-full bg-dark-700 rounded-retro h-2 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      conversion.status === 'validating'
                        ? 'bg-neon-purple animate-pulse-neon'
                        : 'bg-neon-cyan animate-pulse-neon'
                    }`}
                    style={{ width: `${Math.max(conversion.progress, 5)}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs font-mono">
                  <span className={conversion.status === 'validating' ? 'text-neon-purple' : 'text-neon-cyan'}>
                    {conversion.status === 'validating' ? 'Validating output...' : 'Converting to H.264/800x480...'}
                  </span>
                  <span className="text-metal-silver">
                    {Math.round(conversion.progress)}%
                  </span>
                </div>
              </div>
            )}

            {/* Error Message */}
            {conversion.status === 'failed' && conversion.error_message && (
              <div className="mt-2 p-2 bg-neon-orange bg-opacity-10 border border-neon-orange border-opacity-30 rounded">
                <p className="text-xs font-mono text-neon-orange">
                  Error: {conversion.error_message}
                </p>
              </div>
            )}

            {/* Completion Message */}
            {conversion.status === 'completed' && (
              <div className="mt-2 p-2 bg-neon-green bg-opacity-10 border border-neon-green border-opacity-30 rounded">
                <p className="text-xs font-mono text-neon-green">
                  ✓ Converted and added to playlist
                </p>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-metal-silver border-opacity-20">
        <div className="flex justify-between text-xs font-mono text-metal-silver">
          <span>
            Active: {conversions.filter(c => ['queued', 'processing', 'validating'].includes(c.status)).length}
          </span>
          <span>
            Completed: {conversions.filter(c => c.status === 'completed').length}
          </span>
          <span>
            Failed: {conversions.filter(c => c.status === 'failed').length}
          </span>
        </div>
      </div>
    </div>
  )
}

export default ConversionProgress
