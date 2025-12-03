import React, { useState, useEffect } from 'react'
import {
  SunIcon,
  CpuChipIcon,
  WifiIcon,
  ServerIcon,
  PowerIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
  SignalIcon,
  ShieldCheckIcon,
  ArrowPathIcon,
  SpeakerWaveIcon,
  MicrophoneIcon,
  DocumentIcon
} from '@heroicons/react/24/outline'
import { useWebSocket } from '../hooks/useWebSocket'

const SettingsPage = () => {
  const { settings, isConnected } = useWebSocket()
  const [localSettings, setLocalSettings] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null)

  // Network state
  const [networkStatus, setNetworkStatus] = useState(null)
  const [wifiNetworks, setWifiNetworks] = useState([])
  const [selectedNetwork, setSelectedNetwork] = useState(null)
  const [wifiPassword, setWifiPassword] = useState('')
  const [networkLoading, setNetworkLoading] = useState(false)
  const [networkMessage, setNetworkMessage] = useState(null)
  const [scanning, setScanning] = useState(false)

  // Audio state
  const [audioSource, setAudioSource] = useState({ useTestFile: true })
  const [audioLoading, setAudioLoading] = useState(false)
  const [audioMessage, setAudioMessage] = useState(null)

  useEffect(() => {
    fetchSettings()
    fetchNetworkStatus()
    fetchAudioSource()
  }, [])

  const fetchSettings = async () => {
    try {
      const response = await fetch('/api/settings')
      if (response.ok) {
        const data = await response.json()
        setLocalSettings(data)
      }
    } catch (error) {
      console.error('Failed to fetch settings:', error)
    }
  }

  const updateSetting = (key, value) => {
    setLocalSettings(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const updateNestedSetting = (parentKey, childKey, value) => {
    setLocalSettings(prev => ({
      ...prev,
      [parentKey]: {
        ...prev[parentKey],
        [childKey]: value
      }
    }))
  }

  const saveSettings = async () => {
    if (!localSettings) return

    setSaving(true)
    setSaveStatus(null)

    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(localSettings)
      })

      if (response.ok) {
        setSaveStatus({ type: 'success', message: 'Settings saved successfully' })
      } else {
        throw new Error('Failed to save settings')
      }
    } catch (error) {
      setSaveStatus({ type: 'error', message: error.message })
    } finally {
      setSaving(false)
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }

  const setBrightness = async (brightness) => {
    try {
      const response = await fetch('/api/settings/brightness', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(brightness)
      })

      if (response.ok) {
        updateSetting('brightness', brightness)
      }
    } catch (error) {
      console.error('Failed to set brightness:', error)
    }
  }


  // Network management functions
  const fetchNetworkStatus = async () => {
    try {
      const response = await fetch('/api/network/status')
      if (response.ok) {
        const status = await response.json()
        setNetworkStatus(status)
      }
    } catch (error) {
      console.error('Failed to fetch network status:', error)
    }
  }

  const scanWifiNetworks = async () => {
    setScanning(true)
    try {
      const response = await fetch('/api/network/scan')
      if (response.ok) {
        const networks = await response.json()
        setWifiNetworks(networks)
      }
    } catch (error) {
      console.error('Failed to scan WiFi networks:', error)
      setNetworkMessage({ type: 'error', message: 'Failed to scan WiFi networks' })
    } finally {
      setScanning(false)
    }
  }

  const connectToWifi = async (ssid, password) => {
    setNetworkLoading(true)
    setNetworkMessage(null)
    try {
      const response = await fetch('/api/network/connect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ssid, password })
      })

      if (response.ok) {
        setNetworkMessage({ type: 'success', message: `Connected to ${ssid}` })
        setSelectedNetwork(null)
        setWifiPassword('')
        await fetchNetworkStatus()
      } else {
        const error = await response.json()
        setNetworkMessage({ type: 'error', message: error.detail || 'Failed to connect' })
      }
    } catch (error) {
      console.error('Failed to connect to WiFi:', error)
      setNetworkMessage({ type: 'error', message: 'Failed to connect to WiFi' })
    } finally {
      setNetworkLoading(false)
    }
  }

  const enableApMode = async () => {
    setNetworkLoading(true)
    setNetworkMessage(null)
    try {
      const response = await fetch('/api/network/ap/enable', {
        method: 'POST'
      })

      if (response.ok) {
        setNetworkMessage({ type: 'success', message: 'AP mode enabled (prismatron)' })
        await fetchNetworkStatus()
      } else {
        const error = await response.json()
        setNetworkMessage({ type: 'error', message: error.detail || 'Failed to enable AP mode' })
      }
    } catch (error) {
      console.error('Failed to enable AP mode:', error)
      setNetworkMessage({ type: 'error', message: 'Failed to enable AP mode' })
    } finally {
      setNetworkLoading(false)
    }
  }

  const disableApMode = async () => {
    setNetworkLoading(true)
    setNetworkMessage(null)
    try {
      const response = await fetch('/api/network/ap/disable', {
        method: 'POST'
      })

      if (response.ok) {
        setNetworkMessage({ type: 'success', message: 'AP mode disabled' })
        await fetchNetworkStatus()
      } else {
        const error = await response.json()
        setNetworkMessage({ type: 'error', message: error.detail || 'Failed to disable AP mode' })
      }
    } catch (error) {
      console.error('Failed to disable AP mode:', error)
      setNetworkMessage({ type: 'error', message: 'Failed to disable AP mode' })
    } finally {
      setNetworkLoading(false)
    }
  }

  const handleNetworkSelect = (network) => {
    setSelectedNetwork(network)
    setWifiPassword('')
    setNetworkMessage(null)
  }

  const handleWifiConnect = () => {
    if (selectedNetwork) {
      connectToWifi(selectedNetwork.ssid, wifiPassword)
    }
  }

  // Audio source management functions
  const fetchAudioSource = async () => {
    try {
      const response = await fetch('/api/settings/audio-source')
      if (response.ok) {
        const data = await response.json()
        setAudioSource({ useTestFile: data.use_test_file })
      }
    } catch (error) {
      console.error('Failed to fetch audio source:', error)
    }
  }

  const setAudioSourceMode = async (useTestFile) => {
    setAudioLoading(true)
    setAudioMessage(null)
    try {
      const response = await fetch('/api/settings/audio-source', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ use_test_file: useTestFile })
      })

      if (response.ok) {
        setAudioSource({ useTestFile })
        const sourceName = useTestFile ? 'Test File' : 'Live Microphone'
        setAudioMessage({ type: 'success', message: `Audio source changed to ${sourceName}` })
        setTimeout(() => setAudioMessage(null), 3000)
      } else {
        const error = await response.json()
        setAudioMessage({ type: 'error', message: error.detail || 'Failed to change audio source' })
      }
    } catch (error) {
      console.error('Failed to set audio source:', error)
      setAudioMessage({ type: 'error', message: 'Failed to change audio source' })
    } finally {
      setAudioLoading(false)
    }
  }

  // System management functions
  const handleRestart = async () => {
    if (!confirm('Are you sure you want to restart the application? The system will be unavailable for a few seconds.')) {
      return
    }

    try {
      setSaveStatus({ type: 'info', message: 'Restarting application...' })

      const response = await fetch('/api/system/restart', {
        method: 'POST'
      })

      if (response.ok) {
        const data = await response.json()
        setSaveStatus({ type: 'info', message: data.message || 'Application restarting...' })

        // Start polling for reconnection after a delay
        setTimeout(() => {
          pollForReconnection()
        }, 5000)
      } else {
        throw new Error('Failed to restart application')
      }
    } catch (error) {
      console.error('Failed to restart:', error)
      setSaveStatus({ type: 'error', message: 'Failed to restart application' })
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }

  const handleReboot = async () => {
    if (!confirm('Are you sure you want to reboot the device? The system will be unavailable for about a minute.')) {
      return
    }

    try {
      setSaveStatus({ type: 'info', message: 'Rebooting device...' })

      const response = await fetch('/api/system/reboot', {
        method: 'POST'
      })

      if (response.ok) {
        setSaveStatus({ type: 'info', message: 'Device is rebooting. Please wait...' })
      } else {
        throw new Error('Failed to reboot device')
      }
    } catch (error) {
      console.error('Failed to reboot:', error)
      setSaveStatus({ type: 'error', message: 'Failed to reboot device' })
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }

  const pollForReconnection = async () => {
    let attempts = 0
    const maxAttempts = 30 // Try for up to 30 seconds

    const checkConnection = async () => {
      try {
        const response = await fetch('/api/status', {
          method: 'GET',
          cache: 'no-cache'
        })

        if (response.ok) {
          setSaveStatus({ type: 'success', message: 'Application restarted successfully' })
          setTimeout(() => {
            setSaveStatus(null)
            window.location.reload() // Reload the page to reconnect WebSocket
          }, 2000)
          return true
        }
      } catch (error) {
        // Still connecting
      }

      attempts++
      if (attempts < maxAttempts) {
        setTimeout(checkConnection, 1000) // Check every second
      } else {
        setSaveStatus({ type: 'error', message: 'Failed to reconnect after restart. Please refresh the page.' })
      }
      return false
    }

    checkConnection()
  }

  const currentSettings = settings || localSettings

  if (!currentSettings) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="retro-spinner w-8 h-8" />
        <span className="ml-3 text-neon-cyan font-mono">LOADING SETTINGS...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-2xl font-retro font-bold text-neon-cyan text-neon">
          SYSTEM SETTINGS
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          CONFIGURE PRISMATRON DISPLAY
        </p>

        {/* Connection Status */}
        <div className={`inline-flex items-center gap-2 mt-2 px-3 py-1 rounded-retro text-xs font-mono ${
          isConnected
            ? 'text-neon-green border border-neon-green border-opacity-30 bg-neon-green bg-opacity-5'
            : 'text-neon-orange border border-neon-orange border-opacity-30 bg-neon-orange bg-opacity-5'
        }`}>
          <WifiIcon className="w-4 h-4" />
          {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
        </div>
      </div>

      {/* Display Settings */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-pink mb-4 flex items-center gap-2">
          <SunIcon className="w-5 h-5" />
          DISPLAY SETTINGS
        </h3>

        <div className="space-y-6">
          {/* Brightness Control */}
          <div>
            <label className="block text-sm font-retro text-neon-cyan mb-2">
              GLOBAL BRIGHTNESS
            </label>
            <div className="space-y-2">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={currentSettings.brightness}
                onChange={(e) => {
                  const value = parseFloat(e.target.value)
                  setBrightness(value)
                }}
                className="w-full h-2 bg-dark-700 rounded-retro appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #00ffff 0%, #00ffff ${currentSettings.brightness * 100}%, #333 ${currentSettings.brightness * 100}%, #333 100%)`
                }}
              />
              <div className="flex justify-between text-xs font-mono text-metal-silver">
                <span>0%</span>
                <span className="text-neon-cyan">{Math.round(currentSettings.brightness * 100)}%</span>
                <span>100%</span>
              </div>
            </div>
          </div>

          {/* Frame Rate */}
          <div>
            <label className="block text-sm font-retro text-neon-cyan mb-2">
              TARGET FRAME RATE
            </label>
            <select
              value={currentSettings.frame_rate}
              onChange={(e) => updateSetting('frame_rate', parseFloat(e.target.value))}
              className="retro-select w-full"
            >
              <option value={15}>15 FPS (Power Saving)</option>
              <option value={24}>24 FPS (Cinema)</option>
              <option value={30}>30 FPS (Standard)</option>
              <option value={60}>60 FPS (High Performance)</option>
            </select>
          </div>

          {/* Preview Toggle */}
          <div className="flex items-center justify-between">
            <label className="text-sm font-retro text-neon-cyan">
              LIVE PREVIEW
            </label>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={currentSettings.preview_enabled}
                onChange={(e) => updateSetting('preview_enabled', e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-dark-700 peer-focus:outline-none rounded-retro peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-neon-cyan after:rounded-retro after:h-5 after:w-5 after:transition-all peer-checked:bg-neon-cyan peer-checked:bg-opacity-30"></div>
            </label>
          </div>
        </div>
      </div>


      {/* Network Settings */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-purple mb-4 flex items-center gap-2">
          <WifiIcon className="w-5 h-5" />
          NETWORK SETTINGS
        </h3>

        <div className="space-y-6">
          {/* Current Network Status */}
          <div>
            <label className="block text-sm font-retro text-neon-cyan mb-2">
              NETWORK STATUS
            </label>
            {networkStatus ? (
              <div className="retro-input bg-dark-700 text-metal-silver">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <WifiIcon className="w-4 h-4" />
                    <span className="font-mono text-sm">
                      {networkStatus.mode.toUpperCase()}{networkStatus.ssid ? ` - ${networkStatus.ssid}` : ''}
                    </span>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs font-mono ${
                    networkStatus.connected
                      ? 'text-neon-green bg-neon-green bg-opacity-10'
                      : 'text-neon-orange bg-neon-orange bg-opacity-10'
                  }`}>
                    {networkStatus.connected ? 'CONNECTED' : 'DISCONNECTED'}
                  </div>
                </div>
                {networkStatus.ip_address && (
                  <div className="mt-2 text-xs text-metal-silver font-mono">
                    IP: {networkStatus.ip_address}
                  </div>
                )}
              </div>
            ) : (
              <div className="retro-input bg-dark-700 text-metal-silver">
                <span className="font-mono text-sm">Loading network status...</span>
              </div>
            )}
          </div>

          {/* AP Mode Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-retro text-neon-cyan">
                ACCESS POINT MODE
              </label>
              <p className="text-xs text-metal-silver font-mono mt-1">
                Create WiFi hotspot: "prismatron" (no password)
              </p>
            </div>
            <div className="flex items-center gap-2">
              {networkStatus?.mode === 'ap' ? (
                <button
                  onClick={disableApMode}
                  disabled={networkLoading}
                  className="retro-button px-3 py-1 text-neon-orange text-sm font-retro font-bold disabled:opacity-50"
                >
                  DISABLE AP
                </button>
              ) : (
                <button
                  onClick={enableApMode}
                  disabled={networkLoading}
                  className="retro-button px-3 py-1 text-neon-purple text-sm font-retro font-bold disabled:opacity-50"
                >
                  ENABLE AP
                </button>
              )}
            </div>
          </div>

          {/* WiFi Client Mode */}
          {networkStatus?.mode !== 'ap' && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-retro text-neon-cyan">
                  WIFI NETWORKS
                </label>
                <button
                  onClick={scanWifiNetworks}
                  disabled={scanning}
                  className="retro-button px-3 py-1 text-neon-cyan text-xs font-retro font-bold disabled:opacity-50 flex items-center gap-1"
                >
                  <ArrowPathIcon className={`w-3 h-3 ${scanning ? 'animate-spin' : ''}`} />
                  {scanning ? 'SCANNING...' : 'SCAN'}
                </button>
              </div>

              {wifiNetworks.length > 0 ? (
                <div className="space-y-2 max-h-48 overflow-y-auto border border-metal-silver border-opacity-30 rounded-retro p-2">
                  {wifiNetworks.map((network, index) => (
                    <div
                      key={index}
                      onClick={() => handleNetworkSelect(network)}
                      className={`p-3 border rounded-retro cursor-pointer transition-colors ${
                        selectedNetwork?.ssid === network.ssid
                          ? 'border-neon-cyan bg-neon-cyan bg-opacity-10'
                          : 'border-metal-silver border-opacity-30 hover:border-neon-cyan hover:border-opacity-50'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <SignalIcon className="w-4 h-4 text-neon-cyan" />
                          <span className="text-sm font-mono text-neon-cyan">{network.ssid}</span>
                          {network.security !== 'open' && (
                            <ShieldCheckIcon className="w-3 h-3 text-neon-orange" />
                          )}
                          {network.connected && (
                            <span className="text-xs text-neon-green font-mono">CONNECTED</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-metal-silver font-mono">
                            {network.signal_strength}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="retro-input bg-dark-700 text-metal-silver text-center py-4">
                  <span className="font-mono text-sm">No networks found. Click SCAN to search.</span>
                </div>
              )}

              {/* WiFi Connection Form */}
              {selectedNetwork && (
                <div className="retro-container border border-neon-cyan border-opacity-30 bg-neon-cyan bg-opacity-5">
                  <h4 className="text-sm font-retro text-neon-cyan mb-3">
                    CONNECT TO: {selectedNetwork.ssid}
                  </h4>
                  {selectedNetwork.security !== 'open' && (
                    <div className="mb-3">
                      <label className="block text-xs font-retro text-neon-cyan mb-1">
                        PASSWORD
                      </label>
                      <input
                        type="password"
                        value={wifiPassword}
                        onChange={(e) => setWifiPassword(e.target.value)}
                        className="retro-input w-full"
                        placeholder="Enter WiFi password"
                      />
                    </div>
                  )}
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => setSelectedNetwork(null)}
                      className="retro-button px-3 py-1 text-metal-silver text-xs font-retro font-bold"
                    >
                      CANCEL
                    </button>
                    <button
                      onClick={handleWifiConnect}
                      disabled={networkLoading || (selectedNetwork.security !== 'open' && !wifiPassword)}
                      className="retro-button px-3 py-1 text-neon-green text-xs font-retro font-bold disabled:opacity-50"
                    >
                      {networkLoading ? 'CONNECTING...' : 'CONNECT'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Network Status Message */}
          {networkMessage && (
            <div className={`retro-container border ${
              networkMessage.type === 'success'
                ? 'border-neon-green border-opacity-50 bg-neon-green bg-opacity-10'
                : 'border-neon-orange border-opacity-50 bg-neon-orange bg-opacity-10'
            }`}>
              <div className="flex items-center gap-3">
                {networkMessage.type === 'success' ? (
                  <InformationCircleIcon className="w-6 h-6 text-neon-green" />
                ) : (
                  <ExclamationTriangleIcon className="w-6 h-6 text-neon-orange" />
                )}
                <p className={`font-mono text-sm ${
                  networkMessage.type === 'success' ? 'text-neon-green' : 'text-neon-orange'
                }`}>
                  {networkMessage.message}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Audio Settings */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-pink mb-4 flex items-center gap-2">
          <SpeakerWaveIcon className="w-5 h-5" />
          AUDIO SETTINGS
        </h3>

        <div className="space-y-6">
          {/* Audio Source Selection */}
          <div>
            <label className="block text-sm font-retro text-neon-cyan mb-2">
              AUDIO SOURCE
            </label>
            <p className="text-xs text-metal-silver font-mono mb-3">
              Select audio input for beat detection and audio-reactive effects
            </p>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setAudioSourceMode(true)}
                disabled={audioLoading}
                className={`p-4 border rounded-retro cursor-pointer transition-colors flex flex-col items-center gap-2 ${
                  audioSource.useTestFile
                    ? 'border-neon-cyan bg-neon-cyan bg-opacity-10 text-neon-cyan'
                    : 'border-metal-silver border-opacity-30 hover:border-neon-cyan hover:border-opacity-50 text-metal-silver'
                } ${audioLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <DocumentIcon className="w-8 h-8" />
                <span className="text-sm font-retro font-bold">TEST FILE</span>
                <span className="text-xs font-mono opacity-70">whereyouare.wav</span>
              </button>
              <button
                onClick={() => setAudioSourceMode(false)}
                disabled={audioLoading}
                className={`p-4 border rounded-retro cursor-pointer transition-colors flex flex-col items-center gap-2 ${
                  !audioSource.useTestFile
                    ? 'border-neon-cyan bg-neon-cyan bg-opacity-10 text-neon-cyan'
                    : 'border-metal-silver border-opacity-30 hover:border-neon-cyan hover:border-opacity-50 text-metal-silver'
                } ${audioLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <MicrophoneIcon className="w-8 h-8" />
                <span className="text-sm font-retro font-bold">MICROPHONE</span>
                <span className="text-xs font-mono opacity-70">USB Audio Device</span>
              </button>
            </div>
          </div>

          {/* Audio Status Message */}
          {audioMessage && (
            <div className={`retro-container border ${
              audioMessage.type === 'success'
                ? 'border-neon-green border-opacity-50 bg-neon-green bg-opacity-10'
                : 'border-neon-orange border-opacity-50 bg-neon-orange bg-opacity-10'
            }`}>
              <div className="flex items-center gap-3">
                {audioMessage.type === 'success' ? (
                  <InformationCircleIcon className="w-6 h-6 text-neon-green" />
                ) : (
                  <ExclamationTriangleIcon className="w-6 h-6 text-neon-orange" />
                )}
                <p className={`font-mono text-sm ${
                  audioMessage.type === 'success' ? 'text-neon-green' : 'text-neon-orange'
                }`}>
                  {audioMessage.message}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* System Settings */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-orange mb-4 flex items-center gap-2">
          <CpuChipIcon className="w-5 h-5" />
          SYSTEM SETTINGS
        </h3>

        <div className="space-y-6">
          {/* LED Count (Read-only) */}
          <div>
            <label className="block text-sm font-retro text-neon-cyan mb-2">
              LED COUNT
            </label>
            <div className="retro-input bg-dark-700 text-metal-silver cursor-not-allowed">
              {currentSettings.led_count} LEDs
            </div>
          </div>

          {/* Display Resolution (Read-only) */}
          <div>
            <label className="block text-sm font-retro text-neon-cyan mb-2">
              DISPLAY RESOLUTION
            </label>
            <div className="retro-input bg-dark-700 text-metal-silver cursor-not-allowed">
              {currentSettings.display_resolution?.width} × {currentSettings.display_resolution?.height} (5:3)
            </div>
          </div>

          {/* Auto-start Playlist */}
          <div className="flex items-center justify-between">
            <label className="text-sm font-retro text-neon-cyan">
              AUTO-START PLAYLIST ON BOOT
            </label>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={currentSettings.auto_start_playlist}
                onChange={(e) => updateSetting('auto_start_playlist', e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-dark-700 peer-focus:outline-none rounded-retro peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-neon-cyan after:rounded-retro after:h-5 after:w-5 after:transition-all peer-checked:bg-neon-cyan peer-checked:bg-opacity-30"></div>
            </label>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-center">
        <button
          onClick={saveSettings}
          disabled={saving}
          className="retro-button px-8 py-3 text-neon-green text-neon font-retro font-bold disabled:text-metal-silver disabled:cursor-not-allowed"
        >
          {saving ? 'SAVING...' : 'SAVE SETTINGS'}
        </button>
      </div>

      {/* Save Status */}
      {saveStatus && (
        <div className={`retro-container border ${
          saveStatus.type === 'success'
            ? 'border-neon-green border-opacity-50 bg-neon-green bg-opacity-10'
            : 'border-neon-orange border-opacity-50 bg-neon-orange bg-opacity-10'
        }`}>
          <div className="flex items-center gap-3">
            {saveStatus.type === 'success' ? (
              <InformationCircleIcon className="w-6 h-6 text-neon-green" />
            ) : (
              <ExclamationTriangleIcon className="w-6 h-6 text-neon-orange" />
            )}
            <p className={`font-mono text-sm ${
              saveStatus.type === 'success' ? 'text-neon-green' : 'text-neon-orange'
            }`}>
              {saveStatus.message}
            </p>
          </div>
        </div>
      )}

      {/* System Information */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-purple mb-4 flex items-center gap-2">
          <ServerIcon className="w-5 h-5" />
          SYSTEM INFORMATION
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm font-mono">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-metal-silver">Software Version:</span>
              <span className="text-neon-cyan">v1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-metal-silver">Hardware Platform:</span>
              <span className="text-neon-cyan">Jetson Orin Nano</span>
            </div>
            <div className="flex justify-between">
              <span className="text-metal-silver">LED Controller:</span>
              <span className="text-neon-cyan">WLED DigiOcta</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-metal-silver">Communication:</span>
              <span className="text-neon-cyan">UDP/WiFi</span>
            </div>
            <div className="flex justify-between">
              <span className="text-metal-silver">Processing:</span>
              <span className="text-neon-cyan">GPU Accelerated</span>
            </div>
            <div className="flex justify-between">
              <span className="text-metal-silver">Display Type:</span>
              <span className="text-neon-cyan">LED Matrix</span>
            </div>
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="retro-container border border-neon-orange border-opacity-50">
        <h3 className="text-lg font-retro text-neon-orange mb-4 flex items-center gap-2">
          <ExclamationTriangleIcon className="w-5 h-5" />
          DANGER ZONE
        </h3>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 bg-dark-800 rounded-retro">
            <div>
              <h4 className="text-sm font-retro text-neon-orange">RESTART</h4>
              <p className="text-xs text-metal-silver font-mono">
                Restart the application
              </p>
            </div>
            <button
              onClick={handleRestart}
              className="retro-button px-4 py-2 text-neon-orange text-sm font-retro font-bold hover:bg-neon-orange hover:bg-opacity-10 transition-colors"
            >
              <PowerIcon className="w-4 h-4 inline mr-2" />
              RESTART
            </button>
          </div>

          <div className="flex items-center justify-between p-3 bg-dark-800 rounded-retro">
            <div>
              <h4 className="text-sm font-retro text-neon-orange">REBOOT</h4>
              <p className="text-xs text-metal-silver font-mono">
                Reboot the system
              </p>
            </div>
            <button
              onClick={handleReboot}
              className="retro-button px-4 py-2 text-neon-orange text-sm font-retro font-bold hover:bg-neon-orange hover:bg-opacity-10 transition-colors"
            >
              <PowerIcon className="w-4 h-4 inline mr-2" />
              REBOOT
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsPage
