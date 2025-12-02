/**
 * Thumbnail Grid Overlay
 * Shows all slides in a 6-column grid for quick navigation
 * Triggered by 'G' key or Grid button
 */

const ThumbnailGrid = ({ slides, currentSlide, onSelect, onClose }) => {
  return (
    <div
      className="thumbnail-grid-overlay fixed inset-0 bg-black/90 z-50 p-8 overflow-y-auto"
      onClick={onClose}
    >
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-white">
          All Slides ({slides.length})
        </h2>
        <button
          onClick={onClose}
          className="text-white text-2xl hover:text-mllavender transition px-4 py-2"
        >
          ✕ Close
        </button>
      </div>

      {/* Grid */}
      <div
        className="grid grid-cols-6 gap-4 max-w-7xl mx-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {slides.map((slide, index) => (
          <div
            key={slide.id}
            onClick={() => onSelect(index)}
            className={`thumbnail cursor-pointer rounded-lg overflow-hidden border-4 transition-all hover:scale-105 ${
              index === currentSlide
                ? 'border-mlpurple ring-4 ring-mlpurple/50'
                : 'border-transparent hover:border-mllavender2'
            }`}
          >
            {/* Thumbnail Preview */}
            <div className="bg-white aspect-beamer p-2 flex flex-col">
              <div className="bg-mllavender3 px-2 py-1 rounded text-xs">
                <div className="font-bold text-mlpurple truncate">
                  {slide.title || `Slide ${slide.id}`}
                </div>
              </div>
              <div className="flex-1 flex items-center justify-center text-xs text-gray-500 mt-1">
                {slide.metadata?.hasPause && <span className="text-mlorange mr-1">⏸</span>}
                {slide.metadata?.hasColumns && <span className="text-mlblue mr-1">⫘</span>}
                {slide.metadata?.hasTikz && <span className="text-mlgreen mr-1">⬡</span>}
              </div>
              <div className="text-center text-xs font-mono text-gray-600 mt-1">
                {slide.id}
              </div>
            </div>

            {/* Current indicator */}
            {index === currentSlide && (
              <div className="bg-mlpurple text-white text-center text-xs py-1 font-bold">
                CURRENT
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-8 max-w-7xl mx-auto bg-white/10 rounded-lg p-4">
        <div className="flex gap-6 text-white text-sm justify-center">
          <div className="flex items-center gap-2">
            <span className="text-mlorange text-lg">⏸</span>
            <span>Has pause reveal</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-mlblue text-lg">⫘</span>
            <span>Two-column layout</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-mlgreen text-lg">⬡</span>
            <span>TikZ diagram</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThumbnailGrid;
