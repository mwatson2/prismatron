import { useState } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2 } from 'lucide-react'
import { postComment } from '../services/supabase'

/**
 * Form for posting new comments or replies
 */
export default function CommentForm({
  pageSlug,
  parentId = null,
  onCommentPosted,
  onCancel,
  isReply = false,
}) {
  const [content, setContent] = useState('')
  const [authorName, setAuthorName] = useState('')
  const [authorEmail, setAuthorEmail] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!content.trim() || !authorName.trim()) {
      setError('Please fill in your name and comment')
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      await postComment({
        pageSlug,
        content: content.trim(),
        authorName: authorName.trim(),
        authorEmail: authorEmail.trim() || undefined,
        parentId,
      })

      // Clear form
      setContent('')
      if (!isReply) {
        // Keep name/email for top-level comments for convenience
      }

      onCommentPosted?.()
    } catch (err) {
      setError('Failed to post comment. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const inputClasses =
    'w-full bg-dark-800 border border-dark-500 rounded px-3 py-2 text-metal-silver text-sm focus:border-neon-cyan focus:outline-none focus:ring-1 focus:ring-neon-cyan/50 transition-colors placeholder:text-dark-400'

  return (
    <motion.form
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      onSubmit={handleSubmit}
      className="space-y-3"
    >
      {/* Author info row - only show for top-level comments or first reply */}
      {!isReply && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <input
            type="text"
            placeholder="Your name *"
            value={authorName}
            onChange={(e) => setAuthorName(e.target.value)}
            className={inputClasses}
            maxLength={50}
            required
          />
          <input
            type="email"
            placeholder="Email (optional, not displayed)"
            value={authorEmail}
            onChange={(e) => setAuthorEmail(e.target.value)}
            className={inputClasses}
            maxLength={100}
          />
        </div>
      )}

      {/* For replies, show name and email */}
      {isReply && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <input
            type="text"
            placeholder="Your name *"
            value={authorName}
            onChange={(e) => setAuthorName(e.target.value)}
            className={inputClasses}
            maxLength={50}
            required
          />
          <input
            type="email"
            placeholder="Email (optional)"
            value={authorEmail}
            onChange={(e) => setAuthorEmail(e.target.value)}
            className={inputClasses}
            maxLength={100}
          />
        </div>
      )}

      {/* Comment textarea */}
      <textarea
        placeholder={isReply ? 'Write a reply...' : 'Join the discussion...'}
        value={content}
        onChange={(e) => setContent(e.target.value)}
        className={`${inputClasses} min-h-[80px] resize-y`}
        maxLength={2000}
        required
      />

      {/* Error message */}
      {error && (
        <p className="text-neon-pink text-xs">{error}</p>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 justify-end">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 text-sm text-dark-300 hover:text-metal-silver transition-colors"
          >
            Cancel
          </button>
        )}
        <button
          type="submit"
          disabled={isSubmitting || !content.trim() || !authorName.trim()}
          className="inline-flex items-center gap-2 px-4 py-1.5 bg-dark-700 hover:bg-dark-600 border border-neon-cyan/50 hover:border-neon-cyan rounded text-sm text-neon-cyan transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSubmitting ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              Posting...
            </>
          ) : (
            <>
              <Send size={14} />
              {isReply ? 'Reply' : 'Post Comment'}
            </>
          )}
        </button>
      </div>
    </motion.form>
  )
}
