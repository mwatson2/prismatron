import React, { useState, useEffect } from 'react'
import {
  DocumentIcon,
  FilmIcon,
  PhotoIcon,
  PlusIcon,
  FolderIcon
} from '@heroicons/react/24/outline'

const MediaPage = () => {
  const [existingFiles, setExistingFiles] = useState([])
  const [loadingFiles, setLoadingFiles] = useState(false)

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

  // Load existing files on component mount
  useEffect(() => {
    loadExistingFiles()
  }, [])

  const loadExistingFiles = async () => {
    setLoadingFiles(true)
    try {
      const response = await fetch('/api/uploads')
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

  const addExistingFileToPlaylist = async (file) => {
    try {
      const response = await fetch(`/api/uploads/${file.id}/add`, {
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-2xl font-retro font-bold text-neon-orange text-neon">
          MEDIA LIBRARY
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          MANAGE EXISTING FILES & ADD TO PLAYLIST
        </p>
      </div>

      {/* Existing Files */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-orange mb-4 flex items-center gap-2">
          <FolderIcon className="w-5 h-5" />
          EXISTING FILES
        </h3>

        {loadingFiles ? (
          <div className="text-center py-8">
            <div className="text-metal-silver font-mono">Loading files...</div>
          </div>
        ) : existingFiles.length > 0 ? (
          <div className="space-y-2">
            {existingFiles.map((file) => {
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

                  <button
                    onClick={() => addExistingFileToPlaylist(file)}
                    className="retro-button p-2 text-neon-orange hover:text-neon-yellow transition-colors"
                    aria-label="Add to playlist"
                  >
                    <PlusIcon className="w-5 h-5" />
                  </button>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-center py-8">
            <FolderIcon className="w-12 h-12 text-metal-silver mx-auto mb-2 opacity-50" />
            <p className="text-metal-silver font-mono text-sm">
              No existing files found in uploads folder
            </p>
          </div>
        )}
      </div>

      {/* Usage Guidelines */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-purple mb-4">MEDIA GUIDELINES</h3>

        <div className="space-y-3 text-sm font-mono text-metal-silver">
          <div className="flex items-start gap-2">
            <span className="text-neon-orange">•</span>
            <span>Click the + button to add files to the active playlist</span>
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
            <span>Upload new files using the Upload page</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MediaPage

