import React, { useState, useEffect } from 'react'
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd'
import {
  DocumentIcon,
  FilmIcon,
  PhotoIcon,
  PlusIcon,
  FolderIcon,
  ArrowsUpDownIcon,
  PencilIcon,
  TrashIcon
} from '@heroicons/react/24/outline'

const MediaPage = () => {
  const [existingFiles, setExistingFiles] = useState([])
  const [uploadFiles, setUploadFiles] = useState([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [loadingUploads, setLoadingUploads] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  const allowedTypes = {
    image: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
    video: ['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v']
  }

  const getFileType = (fileName) => {
    const ext = fileName.split('.').pop()?.toLowerCase()
    for (const [type, extensions] of Object.entries(allowedTypes)) {
      if (extensions.includes(ext)) {
        return type
      }
    }
    return 'unknown'
  }

  const getFileIcon = (type) => {
    switch (type) {
      case 'image':
        return PhotoIcon
      case 'video':
        return FilmIcon
      default:
        return DocumentIcon
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Load existing files and uploads on component mount
  useEffect(() => {
    loadExistingFiles()
    loadUploadFiles()
  }, [])

  const loadExistingFiles = async () => {
    setLoadingFiles(true)
    try {
      const response = await fetch('/api/media')
      if (response.ok) {
        const data = await response.json()
        setExistingFiles(data.files || [])
      }
    } catch (error) {
      console.error('Failed to load existing files:', error)
    } finally {
      setLoadingFiles(false)
    }
  }

  const loadUploadFiles = async () => {
    setLoadingUploads(true)
    try {
      const response = await fetch('/api/uploads')
      if (response.ok) {
        const data = await response.json()
        setUploadFiles(data.files || [])
      }
    } catch (error) {
      console.error('Failed to load upload files:', error)
    } finally {
      setLoadingUploads(false)
    }
  }

  const addExistingFileToPlaylist = async (file) => {
    try {
      const response = await fetch(`/api/media/${file.id}/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: file.name
        })
      })

      if (response.ok) {
        console.log(`Added ${file.name} to playlist`)
        // Could show a success message here
      }
    } catch (error) {
      console.error('Failed to add file to playlist:', error)
    }
  }

  const moveFileToMedia = async (file) => {
    try {
      const response = await fetch(`/api/uploads/${file.id}/move-to-media`, {
        method: 'POST'
      })

      if (response.ok) {
        console.log(`Moved ${file.name} to media folder`)
        // Reload both lists
        loadExistingFiles()
        loadUploadFiles()
      } else {
        console.error('Failed to move file to media')
      }
    } catch (error) {
      console.error('Failed to move file to media:', error)
    }
  }

  // Drag and drop handlers for react-beautiful-dnd
  const handleDragEnd = async (result) => {
    setIsDragging(false)

    if (!result.destination) return

    // Check if dragging from uploads to media
    if (result.source.droppableId === 'uploads' && result.destination.droppableId === 'media') {
      const draggedFile = uploadFiles[result.source.index]
      if (draggedFile) {
        // Optimistically remove the file from uploads immediately
        setUploadFiles(prev => prev.filter((_, index) => index !== result.source.index))

        await moveFileToMedia(draggedFile)
      }
    }
  }

  const handleRenameFile = async (file) => {
    const newName = window.prompt(`Rename "${file.name}" to:`, file.name)

    if (!newName || newName === file.name) {
      return
    }

    try {
      const response = await fetch(`/api/media/${file.id}/rename`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          new_name: newName
        })
      })

      if (response.ok) {
        const data = await response.json()
        console.log(`Renamed ${file.name} to ${data.new_name}`)
        console.log(`Updated ${data.items_modified} items in ${data.affected_playlists} playlists`)

        // Reload the file list
        loadExistingFiles()
      } else {
        const error = await response.json()
        console.error('Failed to rename file:', error)
        alert(`Failed to rename file: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to rename file:', error)
      alert(`Failed to rename file: ${error.message}`)
    }
  }

  const handleDeleteFile = async (file) => {
    if (!window.confirm(
      `Delete "${file.name}"?\n\nThis will remove the file and all references to it in playlists.\n\nThis action cannot be undone.`
    )) {
      return
    }

    try {
      const response = await fetch(`/api/media/${file.id}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        const data = await response.json()
        console.log(`Deleted ${data.filename}`)
        console.log(`Removed ${data.items_removed} items from ${data.affected_playlists} playlists`)

        // Show success message
        alert(`File deleted successfully!\n\nRemoved ${data.items_removed} items from ${data.affected_playlists} playlist(s).`)

        // Reload the file list
        loadExistingFiles()
      } else {
        const error = await response.json()
        console.error('Failed to delete file:', error)
        alert(`Failed to delete file: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to delete file:', error)
      alert(`Failed to delete file: ${error.message}`)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-2xl font-retro font-bold text-neon-orange text-neon">
          MEDIA LIBRARY
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          MANAGE FILES & ADD TO PLAYLIST
        </p>
      </div>

      {/* Two Column Layout with Drag and Drop */}
      <DragDropContext
        onDragEnd={handleDragEnd}
        onDragStart={() => setIsDragging(true)}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Media Panel (Drop Zone) - First */}
          <div className="retro-container">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-retro text-neon-orange flex items-center gap-2">
                <FolderIcon className="w-5 h-5" />
                MEDIA (PERMANENT)
              </h3>
              <div className="flex items-center gap-2 text-xs text-metal-silver font-mono">
                <ArrowsUpDownIcon className="w-4 h-4" />
                Drag uploads here
              </div>
            </div>

            <Droppable droppableId="media">
              {(provided, snapshot) => (
                <div
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                  className={`space-y-2 min-h-[200px] ${
                    snapshot.isDraggingOver ? 'bg-neon-yellow bg-opacity-10 rounded-retro border-2 border-neon-yellow border-dashed' : ''
                  }`}
                >
                  {loadingFiles ? (
                    <div className="text-center py-8">
                      <div className="text-metal-silver font-mono">Loading files...</div>
                    </div>
                  ) : existingFiles.length > 0 ? (
                    existingFiles.map((file, index) => {
                      const FileIcon = getFileIcon(file.type)
                      return (
                        <div
                          key={file.id}
                          className="flex items-center gap-3 p-3 rounded-retro border border-neon-orange border-opacity-30 bg-neon-orange bg-opacity-5"
                        >
                          <FileIcon className="w-8 h-8 text-neon-orange" />

                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-neon-orange truncate">
                              {file.name}
                            </p>
                            <div className="flex items-center gap-4 text-xs text-metal-silver font-mono">
                              <span className="uppercase">{file.type}</span>
                              <span>{formatFileSize(file.size)}</span>
                              <span>{new Date(file.modified * 1000).toLocaleDateString()}</span>
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => handleRenameFile(file)}
                              className="retro-button p-2 text-neon-purple hover:text-neon-yellow transition-colors"
                              aria-label="Rename file"
                              title="Rename file"
                            >
                              <PencilIcon className="w-5 h-5" />
                            </button>
                            <button
                              onClick={() => handleDeleteFile(file)}
                              className="retro-button p-2 text-red-500 hover:text-red-400 transition-colors"
                              aria-label="Delete file"
                              title="Delete file"
                            >
                              <TrashIcon className="w-5 h-5" />
                            </button>
                            <button
                              onClick={() => addExistingFileToPlaylist(file)}
                              className="retro-button p-2 text-neon-orange hover:text-neon-yellow transition-colors"
                              aria-label="Add to playlist"
                              title="Add to playlist"
                            >
                              <PlusIcon className="w-5 h-5" />
                            </button>
                          </div>
                        </div>
                      )
                    })
                  ) : (
                    <div className="text-center py-8">
                      <FolderIcon className="w-12 h-12 text-metal-silver mx-auto mb-2 opacity-50" />
                      <p className="text-metal-silver font-mono text-sm">
                        {snapshot.isDraggingOver ? 'Drop files here to move to media folder' : 'No files in media folder'}
                      </p>
                    </div>
                  )}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </div>

          {/* Uploads Panel - Second */}
          <div className="retro-container">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-retro text-neon-purple flex items-center gap-2">
                <FolderIcon className="w-5 h-5" />
                UPLOADS (TEMPORARY)
              </h3>
              <div className="text-xs text-metal-silver font-mono">
                Auto-cleanup enabled
              </div>
            </div>

            <Droppable droppableId="uploads">
              {(provided, snapshot) => (
                <div
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                  className="space-y-2 min-h-[200px]"
                >
                  {loadingUploads ? (
                    <div className="text-center py-8">
                      <div className="text-metal-silver font-mono">Loading uploads...</div>
                    </div>
                  ) : uploadFiles.length > 0 ? (
                    <>
                      {uploadFiles.map((file, index) => {
                        const FileIcon = getFileIcon(file.type)
                        return (
                          <Draggable key={file.id} draggableId={file.id} index={index}>
                            {(provided, snapshot) => (
                              <div
                                ref={provided.innerRef}
                                {...provided.draggableProps}
                                className={`flex items-center gap-3 p-3 rounded-retro border border-neon-purple border-opacity-30 bg-neon-purple bg-opacity-5 ${
                                  snapshot.isDragging ? 'shadow-neon-lg scale-105 z-50' : 'cursor-grab active:cursor-grabbing'
                                }`}
                              >
                                {/* Drag Handle */}
                                <div
                                  {...provided.dragHandleProps}
                                  className="cursor-grab active:cursor-grabbing text-metal-silver hover:text-neon-purple"
                                >
                                  <ArrowsUpDownIcon className="w-5 h-5" />
                                </div>

                                <FileIcon className="w-8 h-8 text-neon-purple" />

                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium text-neon-purple truncate">
                                    {file.name}
                                  </p>
                                  <div className="flex items-center gap-4 text-xs text-metal-silver font-mono">
                                    <span className="uppercase">{file.type}</span>
                                    <span>{formatFileSize(file.size)}</span>
                                    <span>{new Date(file.modified * 1000).toLocaleDateString()}</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </Draggable>
                        )
                      })}
                      <div className="text-center py-2 text-xs text-metal-silver font-mono opacity-75">
                        Drag files to Media panel to make them permanent
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-8">
                      <FolderIcon className="w-12 h-12 text-metal-silver mx-auto mb-2 opacity-50" />
                      <p className="text-metal-silver font-mono text-sm">
                        No files in uploads folder
                      </p>
                    </div>
                  )}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </div>
        </div>
      </DragDropContext>

      {/* Usage Guidelines */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-purple mb-4">MEDIA GUIDELINES</h3>

        <div className="space-y-3 text-sm font-mono text-metal-silver">
          <div className="flex items-start gap-2">
            <span className="text-neon-orange">•</span>
            <span>Drag files from Uploads panel to Media panel to make them permanent</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-orange">•</span>
            <span>Click the + button to add media files to the active playlist</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-orange">•</span>
            <span>Files are processed and optimized for LED display</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-orange">•</span>
            <span>Supported formats: JPG, PNG, GIF, MP4, MOV, WEBM</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-orange">•</span>
            <span>Upload new files using the Upload page - uploads folder has auto-cleanup</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MediaPage
