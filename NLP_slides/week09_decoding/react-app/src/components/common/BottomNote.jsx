/**
 * BottomNote Component
 * Recreates LaTeX \bottomnote{} command
 * Appears at bottom of slides with horizontal rule separator
 */

const BottomNote = ({ children }) => {
  if (!children) return null;

  return (
    <div className="bottom-note mt-auto pt-2">
      {/* Horizontal rule - matches \textcolor{mllavender2}{\rule{\textwidth}{0.4pt}} */}
      <div className="w-full h-px bg-mllavender2 mb-2" />

      {/* Content - \footnotesize\textbf{} */}
      <p className="text-[11pt] font-bold text-gray-700 leading-snug">
        {children}
      </p>
    </div>
  );
};

export default BottomNote;
