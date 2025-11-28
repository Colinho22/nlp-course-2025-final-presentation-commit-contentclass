/**
 * Markdown Parser Utility
 * Converts markdown formatting from LaTeX extraction to React elements
 * Handles: **bold**, *italic*, `code`, inline math $...$
 */

import React from 'react';
import { InlineMath } from 'react-katex';

const katexMacros = {
  "\\given": "\\mid",
  "\\argmax": "\\operatorname*{argmax}",
  "\\argmin": "\\operatorname*{argmin}",
  "\\softmax": "\\operatorname{softmax}",
};

/**
 * Parse markdown-formatted text to React elements
 * @param {string} text - Text with markdown formatting
 * @returns {React.ReactNode} - Parsed React elements
 */
export const parseMarkdown = (text) => {
  if (!text || typeof text !== 'string') return text;

  const elements = [];
  let currentIndex = 0;
  let keyCounter = 0;

  // Pattern to match: **bold**, *italic*, `code`, or $math$
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)/g;

  let match;
  while ((match = pattern.exec(text)) !== null) {
    // Add text before match
    if (match.index > currentIndex) {
      elements.push(text.substring(currentIndex, match.index));
    }

    const matched = match[0];

    // Parse the matched formatting
    if (matched.startsWith('**') && matched.endsWith('**')) {
      // Bold
      const content = matched.slice(2, -2);
      elements.push(
        <strong key={`bold-${keyCounter++}`} className="font-bold text-gray-900">
          {content}
        </strong>
      );
    } else if (matched.startsWith('*') && matched.endsWith('*') && !matched.startsWith('**')) {
      // Italic
      const content = matched.slice(1, -1);
      elements.push(
        <em key={`italic-${keyCounter++}`} className="italic">
          {content}
        </em>
      );
    } else if (matched.startsWith('`') && matched.endsWith('`')) {
      // Code
      const content = matched.slice(1, -1);
      elements.push(
        <code key={`code-${keyCounter++}`} className="bg-lightgray px-2 py-0.5 rounded font-mono text-sm">
          {content}
        </code>
      );
    } else if (matched.startsWith('$') && matched.endsWith('$')) {
      // Inline math
      const content = matched.slice(1, -1);
      elements.push(
        <InlineMath key={`math-${keyCounter++}`} math={content} macros={katexMacros} />
      );
    }

    currentIndex = match.index + matched.length;
  }

  // Add remaining text
  if (currentIndex < text.length) {
    elements.push(text.substring(currentIndex));
  }

  // If no formatting found, return original text
  if (elements.length === 0) return text;

  // If only one element and it's a string, return it directly
  if (elements.length === 1 && typeof elements[0] === 'string') {
    return elements[0];
  }

  // Return array of elements (React will handle)
  return elements;
};

/**
 * Component wrapper for markdown parsing
 */
export const MarkdownText = ({ children, className = "" }) => {
  const parsed = parseMarkdown(children);

  return (
    <span className={className}>
      {parsed}
    </span>
  );
};

/**
 * Parse list items with markdown
 */
export const parseListItems = (items) => {
  return items.map((item, index) => ({
    ...item,
    parsedContent: parseMarkdown(typeof item === 'string' ? item : item.content || item)
  }));
};

export default { parseMarkdown, MarkdownText, parseListItems };
