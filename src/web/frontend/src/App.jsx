import React, { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Navigation from './components/Navigation'
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import MediaPage from './pages/MediaPage'
import EffectsPage from './pages/EffectsPage'
import PlaylistPage from './pages/PlaylistPage'
import SettingsPage from './pages/SettingsPage'
import { WebSocketProvider } from './hooks/useWebSocket'
import './App.css'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState(null)

  useEffect(() => {
    // Check if we're running in development or production
    const apiBase = import.meta.env.DEV ? 'http://localhost:8000' : ''
    let interval = null

    // Test API connection
    const checkConnection = async () => {
      // Skip health check if page is hidden to save battery
      if (document.hidden) {
        return
      }

      try {
        const response = await fetch(`${apiBase}/api/health`)
        if (response.ok) {
          setIsConnected(true)
        }
      } catch (error) {
        console.warn('API connection failed:', error)
        setIsConnected(false)
      }
    }

    const startHealthCheck = () => {
      checkConnection()
      interval = setInterval(checkConnection, 30000)
    }

    const stopHealthCheck = () => {
      if (interval) {
        clearInterval(interval)
        interval = null
      }
    }

    // Page Visibility API to pause/resume health checks
    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopHealthCheck()
      } else {
        startHealthCheck()
      }
    }

    // Start initial health check
    startHealthCheck()

    // Listen for visibility changes
    document.addEventListener('visibilitychange', handleVisibilityChange)

    // Safari mobile support
    window.addEventListener('pagehide', stopHealthCheck)
    window.addEventListener('pageshow', (event) => {
      if (!event.persisted || !interval) {
        startHealthCheck()
      }
    })

    return () => {
      stopHealthCheck()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('pagehide', stopHealthCheck)
      window.removeEventListener('pageshow', startHealthCheck)
    }
  }, [])

  return (
    <WebSocketProvider>
      <div className="min-h-screen bg-dark-900 grid-bg">
        {/* Connection status indicator */}
        <div className="fixed top-2 right-2 z-50">
          <div className={`w-3 h-3 rounded-full ${
            isConnected
              ? 'bg-neon-green shadow-neon animate-pulse-neon'
              : 'bg-metal-silver animate-flicker'
          }`} />
        </div>

        {/* Scan line effect */}
        <div className="fixed top-0 left-0 w-full h-0.5 scan-line pointer-events-none" />

        {/* Main content area */}
        <main className="pb-20 px-4 pt-4">
          <Routes>
            <Route path="/" element={<Navigate to="/home" replace />} />
            <Route path="/home" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/media" element={<MediaPage />} />
            <Route path="/effects" element={<EffectsPage />} />
            <Route path="/playlist" element={<PlaylistPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>

        {/* Bottom navigation */}
        <Navigation />
      </div>
    </WebSocketProvider>
  )
}

export default App
