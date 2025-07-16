import React, { createContext, useContext, useEffect, useState, useCallback } from 'react'

const WebSocketContext = createContext(null)

export const useWebSocket = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState(null)
  const [playlist, setPlaylist] = useState({ items: [], current_index: 0, is_playing: false })
  const [settings, setSettings] = useState(null)
  const [previewData, setPreviewData] = useState(null)

  const connect = useCallback(() => {
    const wsUrl = import.meta.env.DEV
      ? 'ws://localhost:8000/ws'
      : `ws://${window.location.host}/ws`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setSocket(ws)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleMessage(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        setSocket(null)

        // Attempt to reconnect after 3 seconds
        setTimeout(connect, 3000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      // Retry connection after 5 seconds
      setTimeout(connect, 5000)
    }
  }, [])

  const handleMessage = useCallback((data) => {
    switch (data.type) {
      case 'initial_state':
        setPlaylist(data.playlist || playlist)
        setSettings(data.settings || settings)
        break

      case 'playback_state':
        setPlaylist(prev => ({
          ...prev,
          is_playing: data.is_playing,
          current_index: data.current_index ?? prev.current_index
        }))
        break

      case 'playlist_position':
        setPlaylist(prev => ({
          ...prev,
          current_index: data.current_index
        }))
        break

      case 'playlist_updated':
        setPlaylist(prev => ({
          ...prev,
          items: data.items || []
        }))
        break

      case 'playlist_state':
        setPlaylist(prev => ({
          ...prev,
          shuffle: data.shuffle ?? prev.shuffle,
          auto_repeat: data.auto_repeat ?? prev.auto_repeat
        }))
        break

      case 'settings_updated':
        setSettings(data.settings)
        break

      case 'brightness_changed':
        setSettings(prev => prev ? { ...prev, brightness: data.brightness } : null)
        break

      case 'system_status':
        setSystemStatus(data)
        break

      case 'preview_data':
        setPreviewData(data)
        break

      default:
        console.log('Unknown WebSocket message type:', data.type)
    }
  }, [playlist, settings])

  const sendMessage = useCallback((message) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, cannot send message:', message)
    }
  }, [socket, isConnected])

  useEffect(() => {
    connect()

    return () => {
      if (socket) {
        socket.close()
      }
    }
  }, [connect])

  const value = {
    isConnected,
    systemStatus,
    playlist,
    settings,
    previewData,
    sendMessage,
    // Convenience methods for common operations
    updatePlaylist: (newPlaylist) => setPlaylist(newPlaylist),
    updateSettings: (newSettings) => setSettings(newSettings),
    updateSystemStatus: (status) => setSystemStatus(status)
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}
