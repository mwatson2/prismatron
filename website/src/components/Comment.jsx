import { useState } from 'react'
import { motion } from 'framer-motion'
import { MessageSquare, ChevronDown, ChevronUp, User } from 'lucide-react'
import CommentForm from './CommentForm'

/**
 * Format a date string to a human-readable format
 */
function formatDate(dateString) {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`

  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  })
}

// Supabase uses snake_case field names
const getAuthorName = (c) => c.author_name || c.authorName || 'Anonymous'
const getCreatedAt = (c) => c.created_at || c.createdAt
const getContent = (c) => c.comment || c.content

/**
 * Individual comment component with nested reply support
 */
export default function Comment({ comment, pageSlug, onCommentPosted, depth = 0 }) {
  const [showReplyForm, setShowReplyForm] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  const maxDepth = 3
  const canReply = depth < maxDepth

  // Get nested replies (built by supabase.js buildCommentTree)
  const replies = comment.replies || []

  const handleReplyPosted = () => {
    setShowReplyForm(false)
    onCommentPosted?.()
  }

  // Calculate left border color based on depth
  const borderColors = [
    'border-neon-cyan',
    'border-neon-pink',
    'border-neon-green',
    'border-neon-purple',
  ]
  const borderColor = borderColors[depth % borderColors.length]

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`${depth > 0 ? 'ml-4 md:ml-6' : ''}`}
    >
      <div
        className={`border-l-2 ${borderColor} pl-4 py-2 bg-dark-800/50 rounded-r`}
      >
        {/* Comment header */}
        <div className="flex items-center gap-2 mb-2">
          <div className="w-6 h-6 rounded-full bg-dark-600 flex items-center justify-center">
            <User size={14} className="text-metal-silver" />
          </div>
          <span className="font-retro text-sm text-neon-cyan">
            {getAuthorName(comment)}
          </span>
          <span className="text-dark-300 text-xs">
            {formatDate(getCreatedAt(comment))}
          </span>

          {/* Collapse button for comments with replies */}
          {replies.length > 0 && (
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="ml-auto p-1 text-dark-300 hover:text-neon-cyan transition-colors"
              title={collapsed ? 'Expand replies' : 'Collapse replies'}
            >
              {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
            </button>
          )}
        </div>

        {/* Comment content */}
        <p className="text-metal-silver text-sm whitespace-pre-wrap break-words">
          {getContent(comment)}
        </p>

        {/* Reply button */}
        {canReply && (
          <button
            onClick={() => setShowReplyForm(!showReplyForm)}
            className="mt-2 flex items-center gap-1 text-xs text-dark-300 hover:text-neon-pink transition-colors"
          >
            <MessageSquare size={14} />
            {showReplyForm ? 'Cancel' : 'Reply'}
          </button>
        )}

        {/* Reply form */}
        {showReplyForm && (
          <div className="mt-3">
            <CommentForm
              pageSlug={pageSlug}
              parentId={comment.id}
              onCommentPosted={handleReplyPosted}
              onCancel={() => setShowReplyForm(false)}
              isReply
            />
          </div>
        )}
      </div>

      {/* Nested replies */}
      {!collapsed && replies.length > 0 && (
        <div className="mt-2 space-y-2">
          {replies.map((reply) => (
            <Comment
              key={reply.id}
              comment={reply}
              pageSlug={pageSlug}
              onCommentPosted={onCommentPosted}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </motion.div>
  )
}
