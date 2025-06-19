import React, { useState } from 'react'
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
  SparklesIcon
} from '@heroicons/react/24/outline'
import { useWebSocket } from '../hooks/useWebSocket'

const PlaylistPage = () => {
  const { playlist } = useWebSocket()
  const [isDragging, setIsDragging] = useState(false)

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
    </div>
  )
}

export default PlaylistPage
