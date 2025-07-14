import React, { useState, useCallback, useRef, useEffect } from 'react'
import {
  CloudArrowUpIcon,
  DocumentIcon,
  FilmIcon,
  PhotoIcon,
  XMarkIcon,
  CheckCircleIcon,
  PlusIcon,
  FolderIcon
} from '@heroicons/react/24/outline'

const UploadPage = () => {
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [existingFiles, setExistingFiles] = useState([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const fileInputRef = useRef(null)

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

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files)
      setSelectedFiles(files)
    }
  }, [])

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const files = Array.from(e.target.files)
      setSelectedFiles(files)
    }
  }

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const uploadFile = async (file, customName = '', duration = null) => {
    const formData = new FormData()
    formData.append('file', file)
    if (customName) formData.append('name', customName)
    if (duration) formData.append('duration', duration.toString())

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const result = await response.json()
      return result
    } catch (error) {
      console.error('Upload error:', error)
      throw error
    }
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return

    setUploading(true)
    setUploadProgress(0)
    setUploadStatus(null)

    try {
      const totalFiles = selectedFiles.length
      let completedFiles = 0

      for (const file of selectedFiles) {
        await uploadFile(file)
        completedFiles++
        setUploadProgress((completedFiles / totalFiles) * 100)
      }

      setUploadStatus({ type: 'success', message: `Successfully uploaded ${totalFiles} file(s)` })
      setSelectedFiles([])

    } catch (error) {
      setUploadStatus({ type: 'error', message: error.message })
    } finally {
      setUploading(false)
      setTimeout(() => {
        setUploadProgress(0)
        setUploadStatus(null)
      }, 3000)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-4">
        <h1 className="text-2xl font-retro font-bold text-neon-cyan text-neon">
          CONTENT UPLOAD
        </h1>
        <p className="text-metal-silver text-sm mt-1 font-mono">
          ADD IMAGES & VIDEOS TO PLAYLIST
        </p>
      </div>

      {/* Upload Area */}
      <div className="retro-container">
        <div
          className={`upload-area ${dragActive ? 'upload-area-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <CloudArrowUpIcon className="w-16 h-16 mx-auto mb-4 text-neon-cyan opacity-60" />

          <div className="space-y-2 text-center">
            <p className="text-lg font-retro text-neon-cyan">
              {dragActive ? 'DROP FILES HERE' : 'DRAG & DROP FILES'}
            </p>
            <p className="text-sm text-metal-silver font-mono">
              or click to browse
            </p>
            <p className="text-xs text-metal-silver">
              Supported: JPG, PNG, GIF, MP4, MOV, WEBM
            </p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.gif,.bmp,.webp,.mp4,.avi,.mov,.mkv,.webm,.m4v"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="retro-container">
          <h3 className="text-lg font-retro text-neon-pink mb-4">SELECTED FILES</h3>

          <div className="space-y-3">
            {selectedFiles.map((file, index) => {
              const fileType = getFileType(file.name)
              const FileIcon = getFileIcon(fileType)
              const isValid = fileType !== 'unknown'

              return (
                <div
                  key={index}
                  className={`flex items-center gap-3 p-3 rounded-retro border ${
                    isValid
                      ? 'border-neon-cyan border-opacity-30 bg-neon-cyan bg-opacity-5'
                      : 'border-neon-orange border-opacity-50 bg-neon-orange bg-opacity-10'
                  }`}
                >
                  <FileIcon className={`w-8 h-8 ${
                    isValid ? 'text-neon-cyan' : 'text-neon-orange'
                  }`} />

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-neon-cyan truncate">
                      {file.name}
                    </p>
                    <div className="flex items-center gap-4 text-xs text-metal-silver font-mono">
                      <span className="uppercase">{fileType}</span>
                      <span>{formatFileSize(file.size)}</span>
                    </div>
                  </div>

                  <button
                    onClick={() => removeFile(index)}
                    className="p-1 text-metal-silver hover:text-neon-orange transition-colors"
                    aria-label="Remove file"
                  >
                    <XMarkIcon className="w-5 h-5" />
                  </button>
                </div>
              )
            })}
          </div>

          {/* Upload Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleUpload}
              disabled={uploading || selectedFiles.every(f => getFileType(f.name) === 'unknown')}
              className="retro-button px-8 py-3 text-neon-green text-neon font-retro font-bold disabled:text-metal-silver disabled:cursor-not-allowed"
            >
              {uploading ? 'UPLOADING...' : 'UPLOAD FILES'}
            </button>
          </div>
        </div>
      )}

      {/* Upload Progress */}
      {uploading && (
        <div className="retro-container">
          <h3 className="text-lg font-retro text-neon-cyan mb-4">UPLOAD PROGRESS</h3>

          <div className="space-y-2">
            <div className="w-full bg-dark-700 rounded-retro h-2 overflow-hidden">
              <div
                className="h-full bg-neon-green transition-all duration-300 animate-pulse-neon"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="text-center text-sm font-mono text-neon-green">
              {Math.round(uploadProgress)}%
            </p>
          </div>
        </div>
      )}

      {/* Upload Status */}
      {uploadStatus && (
        <div className={`retro-container border ${
          uploadStatus.type === 'success'
            ? 'border-neon-green border-opacity-50 bg-neon-green bg-opacity-10'
            : 'border-neon-orange border-opacity-50 bg-neon-orange bg-opacity-10'
        }`}>
          <div className="flex items-center gap-3">
            <CheckCircleIcon className={`w-6 h-6 ${
              uploadStatus.type === 'success' ? 'text-neon-green' : 'text-neon-orange'
            }`} />
            <p className={`font-mono text-sm ${
              uploadStatus.type === 'success' ? 'text-neon-green' : 'text-neon-orange'
            }`}>
              {uploadStatus.message}
            </p>
          </div>
        </div>
      )}

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

      {/* Upload Guidelines */}
      <div className="retro-container">
        <h3 className="text-lg font-retro text-neon-purple mb-4">UPLOAD GUIDELINES</h3>

        <div className="space-y-3 text-sm font-mono text-metal-silver">
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Images will be resized to 800×480 pixels (5:3 ratio)</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Videos will be transcoded for optimal LED display</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Maximum file size: 100MB per file</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span>Files are automatically added to the playlist</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default UploadPage
