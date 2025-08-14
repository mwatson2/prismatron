import React, { useState, useEffect } from 'react'
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd'
import {
  PlayIcon,
  PauseIcon,
  TrashIcon,
  ArrowsUpDownIcon,
  QueueListIcon,
  ArrowPathRoundedSquareIcon,
  ArrowsRightLeftIcon,
  PhotoIcon,
  FilmIcon,
  SparklesIcon,
  Cog6ToothIcon,
  FolderOpenIcon,
  FolderArrowDownIcon,
  DocumentDuplicateIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import { useWebSocket } from '../hooks/useWebSocket'
import TransitionConfig from '../components/TransitionConfig'

const PlaylistPage = () => {
  const { playlist } = useWebSocket()
  const [isDragging, setIsDragging] = useState(false)
  const [transitionConfigItem, setTransitionConfigItem] = useState(null)
  const [savedPlaylists, setSavedPlaylists] = useState([])
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [playlistName, setPlaylistName] = useState('')
  const [playlistDescription, setPlaylistDescription] = useState('')
  const [currentPlaylistFile, setCurrentPlaylistFile] = useState(null)
  const [showSavedPlaylists, setShowSavedPlaylists] = useState(false)

  // Load saved playlists on mount
  useEffect(() => {
    loadSavedPlaylists()
  }, [])

  const loadSavedPlaylists = async () => {
    try {
      const response = await fetch('/api/playlists')
      if (response.ok) {
        const data = await response.json()
        setSavedPlaylists(data.playlists || [])
      }
    } catch (error) {
      console.error('Failed to load saved playlists:', error)
    }
  }

  const loadPlaylist = async (filename) => {
    try {
      const response = await fetch(`/api/playlists/${filename}`)
      if (response.ok) {
        const data = await response.json()
        setCurrentPlaylistFile(filename)
        setShowSavedPlaylists(false)
        // Reload saved playlists to update metadata
        loadSavedPlaylists()
      } else {
        console.error('Failed to load playlist')
      }
    } catch (error) {
      console.error('Failed to load playlist:', error)
    }
  }

  const savePlaylist = async () => {
    if (!playlistName.trim()) return

    try {
      const response = await fetch('/api/playlists/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: playlistName,
          description: playlistDescription,
          overwrite: false
        })
      })

      if (response.ok) {
        const data = await response.json()
        setCurrentPlaylistFile(data.filename)
        setShowSaveDialog(false)
        setPlaylistName('')
        setPlaylistDescription('')
        // Reload saved playlists
        loadSavedPlaylists()
      } else if (response.status === 409) {
        // File exists, ask to overwrite
        if (window.confirm(`Playlist "${playlistName}" already exists. Overwrite?`)) {
          const overwriteResponse = await fetch('/api/playlists/save', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              name: playlistName,
              description: playlistDescription,
              overwrite: true
            })
          })
          if (overwriteResponse.ok) {
            const data = await overwriteResponse.json()
            setCurrentPlaylistFile(data.filename)
            setShowSaveDialog(false)
            setPlaylistName('')
            setPlaylistDescription('')
            loadSavedPlaylists()
          }
        }
      }
    } catch (error) {
      console.error('Failed to save playlist:', error)
    }
  }

  const deletePlaylist = async (filename) => {
    if (!window.confirm(`Delete playlist "${filename}"? This cannot be undone.`)) return

    try {
      const response = await fetch(`/api/playlists/${filename}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        if (currentPlaylistFile === filename) {
          setCurrentPlaylistFile(null)
        }
        // Reload saved playlists
        loadSavedPlaylists()
      } else {
        console.error('Failed to delete playlist')
      }
    } catch (error) {
      console.error('Failed to delete playlist:', error)
    }
  }

  const formatDuration = (seconds) => {
    if (!seconds) return '--:--'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getItemIcon = (type) => {
    switch (type) {
      case 'image':
        return PhotoIcon
      case 'video':
        return FilmIcon
      case 'effect':
        return SparklesIcon
      default:
        return QueueListIcon
    }
  }

  const getItemColor = (type, isActive = false) => {
    const colors = {
      image: isActive ? 'text-neon-cyan' : 'text-neon-cyan',
      video: isActive ? 'text-neon-purple' : 'text-neon-purple',
      effect: isActive ? 'text-neon-pink' : 'text-neon-pink'
    }
    return colors[type] || (isActive ? 'text-neon-cyan' : 'text-metal-silver')
  }

  const handleDragEnd = async (result) => {
    setIsDragging(false)

    if (!result.destination) return

    const items = Array.from(playlist.items || [])
    const [reorderedItem] = items.splice(result.source.index, 1)
    items.splice(result.destination.index, 0, reorderedItem)

    try {
      const response = await fetch('/api/playlist/reorder', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(items.map(item => item.id))
      })

      if (!response.ok) {
        console.error('Failed to reorder playlist')
      }
    } catch (error) {
      console.error('Failed to reorder playlist:', error)
    }
  }

  const removeItem = async (itemId) => {
    try {
      const response = await fetch(`/api/playlist/${itemId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        console.error('Failed to remove item')
      }
    } catch (error) {
      console.error('Failed to remove item:', error)
    }
  }

  const clearPlaylist = async () => {
    if (playlist.items?.length === 0) return

    if (window.confirm('Clear entire playlist? This cannot be undone.')) {
      try {
        const response = await fetch('/api/playlist/clear', {
          method: 'POST'
        })

        if (!response.ok) {
          console.error('Failed to clear playlist')
        }
      } catch (error) {
        console.error('Failed to clear playlist:', error)
      }
    }
  }

  const toggleShuffle = async () => {
    try {
      const response = await fetch('/api/playlist/shuffle', {
        method: 'POST'
      })

      if (!response.ok) {
        console.error('Failed to toggle shuffle')
      }
    } catch (error) {
      console.error('Failed to toggle shuffle:', error)
    }
  }

  const toggleRepeat = async () => {
    try {
      const response = await fetch('/api/playlist/repeat', {
        method: 'POST'
      })

      if (!response.ok) {
        console.error('Failed to toggle repeat')
      }
    } catch (error) {
      console.error('Failed to toggle repeat:', error)
    }
  }

  const jumpToItem = async (index) => {
    // This would require implementing a "jump to" endpoint in the API
    console.log('Jump to item:', index)
  }

  const openTransitionConfig = (item) => {
    setTransitionConfigItem(item)
  }

  const closeTransitionConfig = () => {
    setTransitionConfigItem(null)
  }

  const handleTransitionUpdate = (result) => {
    console.log('Transition updated:', result)
    // The playlist will be updated via WebSocket from the sync service
    closeTransitionConfig()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-2xl font-retro font-bold text-neon-cyan text-neon">
          PLAYLIST MANAGER
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          ORGANIZE & CONTROL PLAYBACK QUEUE
        </p>
        {currentPlaylistFile && (
          <p className="text-neon-green text-xs mt-2 font-mono">
            Loaded: {currentPlaylistFile}
          </p>
        )}
      </div>

      {/* Playlist Controls */}
      <div className="retro-container">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-retro text-neon-pink">PLAYBACK OPTIONS</h3>
          <div className="text-xs font-mono text-metal-silver">
            {playlist.items?.length || 0} items
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => setShowSavedPlaylists(!showSavedPlaylists)}
            className="retro-button px-4 py-2 text-sm font-retro font-bold text-neon-cyan"
          >
            <FolderOpenIcon className="w-4 h-4 inline mr-2" />
            LOAD
          </button>

          <button
            onClick={() => {
              if (currentPlaylistFile) {
                // Quick save to current file
                fetch('/api/playlists/save', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    name: currentPlaylistFile.replace('.json', ''),
                    description: '',
                    overwrite: true
                  })
                }).then(() => loadSavedPlaylists())
              } else {
                setShowSaveDialog(true)
              }
            }}
            disabled={!playlist.items?.length}
            className="retro-button px-4 py-2 text-sm font-retro font-bold text-neon-green disabled:text-metal-silver disabled:cursor-not-allowed"
          >
            <FolderArrowDownIcon className="w-4 h-4 inline mr-2" />
            SAVE
          </button>

          <button
            onClick={() => setShowSaveDialog(true)}
            disabled={!playlist.items?.length}
            className="retro-button px-4 py-2 text-sm font-retro font-bold text-neon-purple disabled:text-metal-silver disabled:cursor-not-allowed"
          >
            <DocumentDuplicateIcon className="w-4 h-4 inline mr-2" />
            SAVE AS
          </button>

          <div className="flex-1" />

          <button
            onClick={toggleShuffle}
            className={`retro-button px-4 py-2 text-sm font-retro font-bold ${
              playlist.shuffle
                ? 'text-neon-orange text-neon border-neon-orange border-opacity-50'
                : 'text-metal-silver'
            }`}
          >
            <ArrowsRightLeftIcon className="w-4 h-4 inline mr-2" />
            SHUFFLE
          </button>

          <button
            onClick={toggleRepeat}
            className={`retro-button px-4 py-2 text-sm font-retro font-bold ${
              playlist.auto_repeat
                ? 'text-neon-green text-neon border-neon-green border-opacity-50'
                : 'text-metal-silver'
            }`}
          >
            <ArrowPathRoundedSquareIcon className="w-4 h-4 inline mr-2" />
            REPEAT
          </button>

          <button
            onClick={clearPlaylist}
            disabled={!playlist.items?.length}
            className="retro-button px-4 py-2 text-sm font-retro font-bold text-neon-orange disabled:text-metal-silver disabled:cursor-not-allowed"
          >
            <TrashIcon className="w-4 h-4 inline mr-2" />
            CLEAR ALL
          </button>
        </div>
      </div>

      {/* Playlist Items */}
      {playlist.items?.length > 0 ? (
        <div className="retro-container">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-retro text-neon-cyan">QUEUE</h3>
            <div className="flex items-center gap-2 text-xs text-metal-silver font-mono">
              <ArrowsUpDownIcon className="w-4 h-4" />
              Drag to reorder
            </div>
          </div>

          <DragDropContext
            onDragEnd={handleDragEnd}
            onDragStart={() => setIsDragging(true)}
          >
            <Droppable droppableId="playlist">
              {(provided, snapshot) => (
                <div
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                  className={`space-y-2 ${
                    snapshot.isDraggingOver ? 'bg-neon-cyan bg-opacity-5 rounded-retro' : ''
                  }`}
                >
                  {playlist.items.map((item, index) => {
                    const isCurrentItem = index === playlist.current_index
                    const ItemIcon = getItemIcon(item.type)
                    const itemColor = getItemColor(item.type, isCurrentItem)

                    return (
                      <Draggable key={item.id} draggableId={item.id} index={index}>
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            className={`playlist-item ${
                              isCurrentItem ? 'playlist-item-active' : ''
                            } ${
                              snapshot.isDragging ? 'shadow-neon-lg scale-105' : ''
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              {/* Drag Handle */}
                              <div
                                {...provided.dragHandleProps}
                                className="cursor-grab active:cursor-grabbing text-metal-silver hover:text-neon-cyan"
                              >
                                <ArrowsUpDownIcon className="w-5 h-5" />
                              </div>

                              {/* Item Icon */}
                              <div className={`${itemColor}`}>
                                <ItemIcon className="w-6 h-6" />
                              </div>

                              {/* Item Info */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="text-sm font-medium text-neon-cyan truncate">
                                    {item.name}
                                  </h4>
                                  {isCurrentItem && (
                                    <div className="flex items-center gap-1">
                                      {playlist.is_playing ? (
                                        <PauseIcon className="w-4 h-4 text-neon-pink animate-pulse-neon" />
                                      ) : (
                                        <PlayIcon className="w-4 h-4 text-neon-pink" />
                                      )}
                                      <span className="text-xs text-neon-pink font-mono">NOW PLAYING</span>
                                    </div>
                                  )}
                                </div>

                                <div className="flex items-center justify-between text-xs text-metal-silver font-mono">
                                  <span className="uppercase">{item.type}</span>
                                  <span>{formatDuration(item.duration)}</span>
                                </div>
                              </div>

                              {/* Item Actions */}
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => jumpToItem(index)}
                                  disabled={isCurrentItem}
                                  className="p-2 text-metal-silver hover:text-neon-cyan disabled:opacity-50 disabled:cursor-not-allowed"
                                  aria-label="Play this item"
                                >
                                  <PlayIcon className="w-4 h-4" />
                                </button>

                                <button
                                  onClick={() => openTransitionConfig(item)}
                                  className="p-2 text-metal-silver hover:text-neon-purple"
                                  aria-label="Configure transitions"
                                >
                                  <Cog6ToothIcon className="w-4 h-4" />
                                </button>

                                <button
                                  onClick={() => removeItem(item.id)}
                                  className="p-2 text-metal-silver hover:text-neon-orange"
                                  aria-label="Remove item"
                                >
                                  <TrashIcon className="w-4 h-4" />
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                      </Draggable>
                    )
                  })}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>
        </div>
      ) : (
        <div className="retro-container text-center py-8">
          <QueueListIcon className="w-16 h-16 mx-auto mb-4 text-metal-silver opacity-50" />
          <h3 className="text-lg font-retro text-metal-silver mb-2">EMPTY PLAYLIST</h3>
          <p className="text-sm text-metal-silver font-mono mb-4">
            Add content using the Upload or Effects tabs
          </p>
        </div>
      )}

      {/* Playlist Statistics */}
      {playlist.items?.length > 0 && (
        <div className="retro-container">
          <h3 className="text-lg font-retro text-neon-green mb-4">STATISTICS</h3>

          <div className="grid grid-cols-2 gap-4 text-sm font-mono">
            <div>
              <span className="text-metal-silver">Total Items:</span>
              <span className="text-neon-cyan ml-2">{playlist.items.length}</span>
            </div>
            <div>
              <span className="text-metal-silver">Total Duration:</span>
              <span className="text-neon-cyan ml-2">
                {formatDuration(
                  playlist.items.reduce((total, item) => total + (item.duration || 0), 0)
                )}
              </span>
            </div>
            <div>
              <span className="text-metal-silver">Images:</span>
              <span className="text-neon-cyan ml-2">
                {playlist.items.filter(item => item.type === 'image').length}
              </span>
            </div>
            <div>
              <span className="text-metal-silver">Videos:</span>
              <span className="text-neon-cyan ml-2">
                {playlist.items.filter(item => item.type === 'video').length}
              </span>
            </div>
            <div>
              <span className="text-metal-silver">Effects:</span>
              <span className="text-neon-cyan ml-2">
                {playlist.items.filter(item => item.type === 'effect').length}
              </span>
            </div>
            <div>
              <span className="text-metal-silver">Current:</span>
              <span className="text-neon-cyan ml-2">
                {playlist.current_index + 1} / {playlist.items.length}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Saved Playlists Section */}
      {showSavedPlaylists && savedPlaylists.length > 0 && (
        <div className="retro-container">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-retro text-neon-green">SAVED PLAYLISTS</h3>
            <button
              onClick={() => setShowSavedPlaylists(false)}
              className="p-1 text-metal-silver hover:text-neon-orange"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {savedPlaylists.map(savedPlaylist => (
              <div
                key={savedPlaylist.filename}
                className="playlist-item flex items-center justify-between"
              >
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-medium text-neon-cyan truncate">
                    {savedPlaylist.name}
                  </h4>
                  <div className="flex items-center gap-3 text-xs text-metal-silver font-mono">
                    <span>{savedPlaylist.item_count} items</span>
                    <span>{formatDuration(savedPlaylist.total_duration)}</span>
                    {savedPlaylist.description && (
                      <span className="truncate italic">{savedPlaylist.description}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => loadPlaylist(savedPlaylist.filename)}
                    className="p-2 text-metal-silver hover:text-neon-cyan"
                    aria-label="Load playlist"
                  >
                    <FolderOpenIcon className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => deletePlaylist(savedPlaylist.filename)}
                    className="p-2 text-metal-silver hover:text-neon-orange"
                    aria-label="Delete playlist"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Save Dialog Modal */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-deep-space bg-opacity-90 flex items-center justify-center z-50">
          <div className="retro-container max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-retro text-neon-cyan">SAVE PLAYLIST</h3>
              <button
                onClick={() => {
                  setShowSaveDialog(false)
                  setPlaylistName('')
                  setPlaylistDescription('')
                }}
                className="p-1 text-metal-silver hover:text-neon-orange"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>

            <input
              type="text"
              value={playlistName}
              onChange={(e) => setPlaylistName(e.target.value)}
              placeholder="Playlist Name"
              className="w-full px-3 py-2 bg-deep-space border border-neon-cyan border-opacity-50 rounded-retro text-neon-cyan font-mono text-sm focus:outline-none focus:border-opacity-100 mb-3"
              autoFocus
            />

            <textarea
              value={playlistDescription}
              onChange={(e) => setPlaylistDescription(e.target.value)}
              placeholder="Description (optional)"
              className="w-full px-3 py-2 bg-deep-space border border-neon-cyan border-opacity-50 rounded-retro text-neon-cyan font-mono text-sm focus:outline-none focus:border-opacity-100 mb-4 resize-none"
              rows={3}
            />

            <div className="flex gap-3">
              <button
                onClick={savePlaylist}
                disabled={!playlistName.trim()}
                className="retro-button flex-1 px-4 py-2 text-sm font-retro font-bold text-neon-green disabled:text-metal-silver disabled:cursor-not-allowed"
              >
                SAVE
              </button>
              <button
                onClick={() => {
                  setShowSaveDialog(false)
                  setPlaylistName('')
                  setPlaylistDescription('')
                }}
                className="retro-button flex-1 px-4 py-2 text-sm font-retro font-bold text-metal-silver"
              >
                CANCEL
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Transition Configuration Modal */}
      {transitionConfigItem && (
        <TransitionConfig
          item={transitionConfigItem}
          onUpdate={handleTransitionUpdate}
          onClose={closeTransitionConfig}
        />
      )}
    </div>
  )
}

export default PlaylistPage
