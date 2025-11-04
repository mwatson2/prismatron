import { useState, useEffect, useCallback } from 'react'
import { useWebSocket } from './useWebSocket'

const useConversions = () => {
  const [conversions, setConversions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const { conversionUpdate } = useWebSocket()

  // Fetch conversions from API
  const fetchConversions = useCallback(async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/conversions')
      if (!response.ok) {
        throw new Error(`Failed to fetch conversions: ${response.statusText}`)
      }
      const data = await response.json()
      setConversions(data.conversions || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching conversions:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  // Cancel a conversion
  const cancelConversion = useCallback(async (jobId) => {
    try {
      const response = await fetch(`/api/conversions/${jobId}/cancel`, {
        method: 'POST'
      })
      if (!response.ok) {
        throw new Error(`Failed to cancel conversion: ${response.statusText}`)
      }
      // Refresh conversions list
      await fetchConversions()
      return true
    } catch (err) {
      console.error('Error cancelling conversion:', err)
      setError(err.message)
      return false
    }
  }, [fetchConversions])

  // Remove a conversion from the list
  const removeConversion = useCallback(async (jobId) => {
    try {
      const response = await fetch(`/api/conversions/${jobId}`, {
        method: 'DELETE'
      })
      if (!response.ok) {
        throw new Error(`Failed to remove conversion: ${response.statusText}`)
      }
      // Refresh conversions list
      await fetchConversions()
      return true
    } catch (err) {
      console.error('Error removing conversion:', err)
      setError(err.message)
      return false
    }
  }, [fetchConversions])

  // Update a specific conversion (for WebSocket updates)
  const updateConversion = useCallback((updatedConversion) => {
    setConversions(prev => {
      const index = prev.findIndex(c => c.id === updatedConversion.id)
      if (index >= 0) {
        // Update existing conversion
        const newConversions = [...prev]
        newConversions[index] = updatedConversion
        return newConversions
      } else {
        // Add new conversion
        return [...prev, updatedConversion]
      }
    })
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchConversions()
  }, [fetchConversions])

  // Subscribe to WebSocket conversion updates
  useEffect(() => {
    if (conversionUpdate && conversionUpdate.conversion) {
      updateConversion(conversionUpdate.conversion)
    }
  }, [conversionUpdate, updateConversion])

  return {
    conversions,
    loading,
    error,
    fetchConversions,
    cancelConversion,
    removeConversion,
    updateConversion
  }
}

export default useConversions
