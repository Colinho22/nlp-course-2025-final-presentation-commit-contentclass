/**
 * Full Navigation Controls
 * Floating control bar with prev/next, progress, and view toggles
 */

const Controls = ({
  onPrev,
  onNext,
  onToggleThumbnails,
  onToggleSections,
  current,
  total,
  currentSection,
  canPrev,
  canNext
}) => {
  return (
    <div className="controls fixed bottom-6 left-1/2 -translate-x-1/2 z-30">
      <div className="bg-white/95 backdrop-blur-sm rounded-full shadow-2xl px-6 py-3 flex items-center gap-4 border-2 border-mllavender2">
        {/* Previous Button */}
        <button
          onClick={onPrev}
          disabled={!canPrev}
          className="btn-nav px-4 py-2 bg-mlpurple text-white rounded-lg font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-opacity-90 transition"
          aria-label="Previous slide"
        >
          ← Prev
        </button>

        {/* Slide Counter */}
        <div className="slide-info text-center px-4">
          <div className="text-lg font-bold text-mlpurple font-mono">
            {current} / {total}
          </div>
          <div className="text-xs text-gray-600 capitalize">
            {currentSection}
          </div>
        </div>

        {/* Next Button */}
        <button
          onClick={onNext}
          disabled={!canNext}
          className="btn-nav px-4 py-2 bg-mlpurple text-white rounded-lg font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-opacity-90 transition"
          aria-label="Next slide"
        >
          Next →
        </button>

        {/* Divider */}
        <div className="border-l-2 border-gray-300 h-10 mx-2" />

        {/* Thumbnail Grid Toggle */}
        <button
          onClick={onToggleThumbnails}
          className="btn-icon px-3 py-2 bg-mllavender3 text-mlpurple rounded-lg hover:bg-mllavender2 transition"
          aria-label="Toggle thumbnail grid (G)"
          title="Thumbnail Grid (G)"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        </button>

        {/* Section Menu Toggle */}
        <button
          onClick={onToggleSections}
          className="btn-icon px-3 py-2 bg-mllavender3 text-mlpurple rounded-lg hover:bg-mllavender2 transition"
          aria-label="Toggle section menu (S)"
          title="Section Menu (S)"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>

      {/* Keyboard Shortcuts Hint */}
      <div className="text-center mt-2 text-xs text-white/70">
        ← → : Navigate | G: Grid | S: Sections | 1-9: Jump to section | Esc: Close
      </div>
    </div>
  );
};

export default Controls;
