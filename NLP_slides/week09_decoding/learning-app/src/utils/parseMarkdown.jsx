/**
 * Simple markdown parser for bold, italic, code formatting
 * Converts **bold**, *italic*, `code` to React elements
 */

export const parseMarkdown = (text) => {
  if (!text || typeof text !== 'string') return text;

  // Split by markdown patterns while preserving them
  const parts = [];
  let remaining = text;

  // Pattern: **bold** or *italic* or `code`
  const regex = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
    }

    const matched = match[0];

    if (matched.startsWith('**') && matched.endsWith('**')) {
      // Bold
      parts.push({ type: 'bold', content: matched.slice(2, -2) });
    } else if (matched.startsWith('*') && matched.endsWith('*')) {
      // Italic
      parts.push({ type: 'italic', content: matched.slice(1, -1) });
    } else if (matched.startsWith('`') && matched.endsWith('`')) {
      // Code
      parts.push({ type: 'code', content: matched.slice(1, -1) });
    }

    lastIndex = match.index + matched.length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.substring(lastIndex) });
  }

  // If no formatting found, return original
  if (parts.length === 0) return text;

  // Convert parts to React elements
  return parts.map((part, i) => {
    switch (part.type) {
      case 'bold':
        return <strong key={i} style={{ fontWeight: 700, color: '#2c2c2c' }}>{part.content}</strong>;
      case 'italic':
        return <em key={i}>{part.content}</em>;
      case 'code':
        return <code key={i} style={{ backgroundColor: '#f0f0f0', padding: '2px 6px', borderRadius: '3px', fontFamily: 'Monaco, monospace' }}>{part.content}</code>;
      default:
        return part.content;
    }
  });
};

export default parseMarkdown;
