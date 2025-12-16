import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { MessageCircle, RefreshCw, Loader2 } from 'lucide-react'
import { getComments, isCommentsEnabled } from '../services/supabase'
import Comment from './Comment'
import CommentForm from './CommentForm'

/**
 * Main comments container component
 * Fetches and displays comments for a specific page with nested replies
 */
export default function Comments({ pageSlug }) {
  // Don't render if Supabase is not configured
  if (!isCommentsEnabled) {
    return null
  }
  const [comments, setComments] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState(null)

  const fetchComments = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) {
      setIsRefreshing(true)
    }
    setError(null)

    try {
      const data = await getComments(pageSlug)
      setComments(data)
    } catch (err) {
      setError('Failed to load comments')
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [pageSlug])

  useEffect(() => {
    fetchComments()
  }, [fetchComments])

  const handleCommentPosted = () => {
    // Refresh comments after posting
    fetchComments(true)
  }

  const handleRefresh = () => {
    fetchComments(true)
  }

  return (
    <section className="content-card mt-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="subsection-title flex items-center gap-2 mb-0">
          <MessageCircle size={20} />
          Discussion
          {comments.length > 0 && (
            <span className="text-sm font-normal text-dark-300">
              ({comments.length})
            </span>
          )}
        </h2>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="p-2 text-dark-300 hover:text-neon-cyan transition-colors disabled:opacity-50"
          title="Refresh comments"
        >
          <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* New comment form */}
      <div className="mb-6 pb-6 border-b border-dark-600">
        <CommentForm
          pageSlug={pageSlug}
          onCommentPosted={handleCommentPosted}
        />
      </div>

      {/* Comments list */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={24} className="animate-spin text-neon-cyan" />
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <p className="text-neon-pink text-sm mb-2">{error}</p>
            <button
              onClick={() => fetchComments()}
              className="text-sm text-neon-cyan hover:underline"
            >
              Try again
            </button>
          </div>
        ) : comments.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8"
          >
            <MessageCircle size={32} className="mx-auto text-dark-400 mb-2" />
            <p className="text-dark-300 text-sm">
              No comments yet. Be the first to start the discussion!
            </p>
          </motion.div>
        ) : (
          comments.map((comment) => (
            <Comment
              key={comment.id}
              comment={comment}
              pageSlug={pageSlug}
              onCommentPosted={handleCommentPosted}
            />
          ))
        )}
      </div>
    </section>
  )
}
