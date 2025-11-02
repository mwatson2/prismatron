import { useState, useEffect, useCallback } from 'react'

const useConversions = () => {
  const [conversions, setConversions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

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

  // Poll for updates every 2 seconds if there are active conversions
  useEffect(() => {
    const hasActiveConversions = conversions.some(c =>
      ['queued', 'processing', 'validating'].includes(c.status)
    )

    if (hasActiveConversions) {
      const interval = setInterval(fetchConversions, 2000)
      return () => clearInterval(interval)
    }
  }, [conversions, fetchConversions])

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
