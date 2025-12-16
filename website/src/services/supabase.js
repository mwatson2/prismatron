/**
 * Supabase client for comments
 */

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY

// Comments feature is disabled if Supabase is not configured
export const isCommentsEnabled = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY)

if (!isCommentsEnabled) {
  console.warn('Comments disabled: VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY not configured')
}

/**
 * Make a request to Supabase REST API
 */
async function supabaseRequest(endpoint, options = {}) {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${endpoint}`, {
    ...options,
    headers: {
      'apikey': SUPABASE_ANON_KEY,
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
      'Content-Type': 'application/json',
      'Prefer': options.method === 'POST' ? 'return=representation' : undefined,
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Supabase error: ${error}`)
  }

  // DELETE returns no content
  if (response.status === 204) {
    return null
  }

  return response.json()
}

/**
 * Build nested comment tree from flat array
 */
function buildCommentTree(comments) {
  const commentMap = new Map()
  const roots = []

  // First pass: create map of all comments
  comments.forEach(comment => {
    commentMap.set(comment.id, { ...comment, replies: [] })
  })

  // Second pass: build tree structure
  comments.forEach(comment => {
    const node = commentMap.get(comment.id)
    if (comment.parent) {
      const parentNode = commentMap.get(comment.parent)
      if (parentNode) {
        parentNode.replies.push(node)
      } else {
        // Parent not found (maybe deleted), treat as root
        roots.push(node)
      }
    } else {
      roots.push(node)
    }
  })

  // Sort replies by created_at (oldest first for conversation flow)
  const sortReplies = (node) => {
    node.replies.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    node.replies.forEach(sortReplies)
  }
  roots.forEach(sortReplies)

  // Sort roots by created_at (newest first)
  roots.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))

  return roots
}

/**
 * Fetch comments for a specific page
 * @param {string} pageSlug - The page identifier (e.g., 'mechanical', 'electrical')
 * @returns {Promise<Array>} - Array of comments with nested replies
 */
export async function getComments(pageSlug) {
  try {
    // Fetch all comments for this page
    const comments = await supabaseRequest(
      `Comments?page_slug=eq.${encodeURIComponent(pageSlug)}&order=created_at.desc`
    )

    // Build nested tree structure
    return buildCommentTree(comments || [])
  } catch (error) {
    console.error('Error fetching comments:', error)
    return []
  }
}

/**
 * Post a new comment
 * @param {Object} comment - Comment data
 * @param {string} comment.pageSlug - The page identifier
 * @param {string} comment.content - Comment content
 * @param {string} comment.authorName - Author's display name
 * @param {string} [comment.authorEmail] - Author's email (optional)
 * @param {string} [comment.parentId] - Parent comment UUID for replies
 * @returns {Promise<Object>} - Created comment
 */
export async function postComment({ pageSlug, content, authorName, authorEmail, parentId }) {
  try {
    const data = await supabaseRequest('Comments', {
      method: 'POST',
      body: JSON.stringify({
        page_slug: pageSlug,
        comment: content,
        author_name: authorName,
        author_email: authorEmail || null,
        parent: parentId || null,
      }),
    })

    return data[0]
  } catch (error) {
    console.error('Error posting comment:', error)
    throw error
  }
}

/**
 * Delete a comment (requires RLS policy or admin key)
 * @param {string} commentId - Comment UUID to delete
 * @returns {Promise<boolean>} - Success status
 */
export async function deleteComment(commentId) {
  try {
    await supabaseRequest(`Comments?id=eq.${commentId}`, {
      method: 'DELETE',
    })
    return true
  } catch (error) {
    console.error('Error deleting comment:', error)
    return false
  }
}
