import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react'

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
  const [currentPlaylistFile, setCurrentPlaylistFile] = useState(null)
  const [playlistModified, setPlaylistModified] = useState(false)
  const [isLoadingPlaylist, setIsLoadingPlaylist] = useState(false)
  const [isPageVisible, setIsPageVisible] = useState(!document.hidden)
  const [conversionUpdate, setConversionUpdate] = useState(null)
  const [audioReactiveTriggersChanged, setAudioReactiveTriggersChanged] = useState(null)
  const [audioConfigSaved, setAudioConfigSaved] = useState(null)

  // Use refs to avoid stale closures in callbacks
  const currentPlaylistFileRef = useRef(currentPlaylistFile)
  const playlistModifiedRef = useRef(playlistModified)
  const isLoadingPlaylistRef = useRef(isLoadingPlaylist)
  const socketRef = useRef(socket)
  const reconnectTimeoutRef = useRef(null)

  useEffect(() => {
    currentPlaylistFileRef.current = currentPlaylistFile
  }, [currentPlaylistFile])

  useEffect(() => {
    playlistModifiedRef.current = playlistModified
  }, [playlistModified])

  useEffect(() => {
    isLoadingPlaylistRef.current = isLoadingPlaylist
  }, [isLoadingPlaylist])

  useEffect(() => {
    socketRef.current = socket
  }, [socket])

  // Page Visibility API with Safari mobile support
  useEffect(() => {
    const handleVisibilityChange = () => {
      const isVisible = !document.hidden
      setIsPageVisible(isVisible)

      if (isVisible) {
        // Page became visible - reconnect if not already connected
        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
          connect()
        }
      } else {
        // Page became hidden - disconnect WebSocket
        disconnect()
      }
    }

    // Standard Page Visibility API
    document.addEventListener('visibilitychange', handleVisibilityChange)

    // Safari mobile support - additional events
    window.addEventListener('pagehide', () => {
      setIsPageVisible(false)
      disconnect()
    })

    window.addEventListener('pageshow', (event) => {
      setIsPageVisible(true)
      if (!event.persisted) {
        // Page was not loaded from cache, normal page load
        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
          connect()
        }
      } else {
        // Page was loaded from cache (back/forward navigation)
        connect()
      }
    })

    // iOS Safari additional support
    window.addEventListener('beforeunload', () => {
      disconnect()
    })

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('pagehide', handleVisibilityChange)
      window.removeEventListener('pageshow', handleVisibilityChange)
      window.removeEventListener('beforeunload', disconnect)
    }
  }, [])

  const disconnect = useCallback(() => {
    // Clear any pending reconnection attempts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.close()
    }
    setSocket(null)
    setIsConnected(false)
  }, [])

  const connect = useCallback(() => {
    // Only connect if page is visible
    if (document.hidden) {
      return
    }

    // Clear any pending reconnection attempts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    const wsUrl = import.meta.env.DEV
      ? 'ws://localhost:8000/ws'
      : `ws://${window.location.host}/ws`

    try {
      const ws = new WebSocket(wsUrl)

      // State for handling multi-part messages with binary fields
      let pendingMessage = null
      let pendingBinaryFields = []
      let receivedBinaryData = []

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setSocket(ws)
      }

      ws.onmessage = (event) => {
        try {
          // Check if this is a binary message
          if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
            // Binary message - must be part of a multi-part message
            if (!pendingMessage) {
              console.error('Received binary WebSocket message without pending JSON message')
              return
            }

            // Convert Blob to ArrayBuffer if needed
            if (event.data instanceof Blob) {
              const reader = new FileReader()
              reader.onload = () => {
                receivedBinaryData.push(new Uint8Array(reader.result))
                processPendingMessage()
              }
              reader.readAsArrayBuffer(event.data)
            } else {
              // Already ArrayBuffer
              receivedBinaryData.push(new Uint8Array(event.data))
              processPendingMessage()
            }
          } else {
            // Text message - parse as JSON
            const data = JSON.parse(event.data)

            // Check if this message expects binary data to follow
            if (data._binary_fields && Array.isArray(data._binary_fields) && data._binary_fields.length > 0) {
              // Store message and wait for binary data
              pendingMessage = data
              pendingBinaryFields = [...data._binary_fields]
              receivedBinaryData = []
            } else {
              // Regular JSON message - handle immediately
              handleMessage(data)
            }
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
          console.error('Raw message data:', event.data)
        }
      }

      // Helper to process pending message once all binary data is received
      const processPendingMessage = () => {
        if (!pendingMessage) return

        // Check if we have all expected binary fields
        if (receivedBinaryData.length === pendingBinaryFields.length) {
          // Augment message with binary data as Uint8Array
          pendingBinaryFields.forEach((fieldName, index) => {
            pendingMessage[fieldName] = receivedBinaryData[index]
          })

          // Clean up internal fields
          delete pendingMessage._binary_fields
          delete pendingMessage._binary_sizes

          // Handle the complete message
          handleMessage(pendingMessage)

          // Reset pending state
          pendingMessage = null
          pendingBinaryFields = []
          receivedBinaryData = []
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        setSocket(null)

        // Only attempt to reconnect if page is still visible
        if (!document.hidden) {
          reconnectTimeoutRef.current = setTimeout(connect, 3000)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      // Only retry if page is still visible
      if (!document.hidden) {
        reconnectTimeoutRef.current = setTimeout(connect, 5000)
      }
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
        console.log('📋 Playlist updated received:', {
          itemCount: data.items?.length || 0,
          currentIndex: data.current_index,
          isPlaying: data.is_playing,
          items: data.items?.map(item => ({ id: item.id, name: item.name, file_path: item.file_path }))
        })

        setPlaylist(prev => {
          // Check if items actually changed (not just playback state)
          const itemsChanged = JSON.stringify(prev.items) !== JSON.stringify(data.items || [])

          console.log('📋 Playlist state update:', {
            itemsChanged,
            prevItemCount: prev.items.length,
            newItemCount: data.items?.length || 0
          })

          // Mark as modified if we have a loaded file and items changed
          // BUT skip if we're in the process of loading a playlist
          if (itemsChanged && currentPlaylistFileRef.current && !isLoadingPlaylistRef.current) {
            setPlaylistModified(true)
          }

          // Clear loading flag after playlist update
          if (isLoadingPlaylistRef.current) {
            setIsLoadingPlaylist(false)
          }

          const newState = {
            ...prev,
            items: data.items || [],
            current_index: data.current_index ?? prev.current_index,
            is_playing: data.is_playing ?? prev.is_playing,
            auto_repeat: data.auto_repeat ?? prev.auto_repeat,
            shuffle: data.shuffle ?? prev.shuffle
          }

          console.log('📋 New playlist state:', {
            itemCount: newState.items.length,
            currentIndex: newState.current_index,
            isPlaying: newState.is_playing
          })

          return newState
        })

        // Update current playlist file if provided in the message
        if (data.current_playlist_file !== undefined) {
          setCurrentPlaylistFile(data.current_playlist_file)
        }
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
        // Track how often preview data arrives
        if (!handleMessage.previewLastLogTime) {
          handleMessage.previewLastLogTime = performance.now()
          handleMessage.previewCount = 1  // Count the first message
          handleMessage.lastFrameCounter = -1
        } else {
          handleMessage.previewCount++
          const now = performance.now()
          if (now - handleMessage.previewLastLogTime >= 5000) {
            console.log(`📡 WebSocket preview_data received ${handleMessage.previewCount} times in last 5s (should be ~150 for 30fps)`)
            handleMessage.previewCount = 1  // Reset to 1 to count the current message in the new window
            handleMessage.previewLastLogTime = now
          }
        }

        // Only update previewData if frame_counter has changed to prevent unnecessary re-renders
        // This prevents React useEffect from triggering on every WebSocket message
        if (data.frame_counter !== handleMessage.lastFrameCounter) {
          handleMessage.lastFrameCounter = data.frame_counter
          setPreviewData(data)
        }
        break

      case 'conversion_update':
        // Handle conversion progress updates
        console.log('Conversion update received:', data)
        setConversionUpdate(data)
        break

      case 'audio_reactive_triggers_changed':
        // Audio reactive trigger configuration changed
        console.log('Audio reactive triggers changed:', data)
        // Notify components that care (AudioReactivePanel can listen for this)
        setAudioReactiveTriggersChanged(data)
        break

      case 'audio_config_saved':
        // Audio config was saved to disk
        console.log('Audio config saved:', data)
        // Use timestamp to ensure state change is detected
        setAudioConfigSaved(data.timestamp || Date.now())
        break

      default:
        console.warn('⚠️ Unknown WebSocket message type:', data.type, data)
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
      // Clear any pending reconnection attempts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }

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
    currentPlaylistFile,
    playlistModified,
    conversionUpdate,
    audioReactiveTriggersChanged,
    audioConfigSaved,
    sendMessage,
    // Convenience methods for common operations
    updatePlaylist: (newPlaylist) => setPlaylist(newPlaylist),
    updateSettings: (newSettings) => setSettings(newSettings),
    updateSystemStatus: (status) => setSystemStatus(status),
    setCurrentPlaylistFile: (filename) => setCurrentPlaylistFile(filename),
    setPlaylistModified: (modified) => setPlaylistModified(modified),
    setIsLoadingPlaylist: (loading) => setIsLoadingPlaylist(loading)
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}
