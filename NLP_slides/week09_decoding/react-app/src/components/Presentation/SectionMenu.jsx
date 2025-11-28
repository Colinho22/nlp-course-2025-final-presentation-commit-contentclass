/**
 * Section Menu Sidebar
 * Quick navigation to presentation sections
 * Triggered by 'S' key or Sections button
 */

const SectionMenu = ({ sections, currentSlide, currentSection, onSelectSection, onClose }) => {
  const sectionInfo = {
    intro: { name: 'Introduction', emoji: '📋', color: 'mlblue' },
    extremes: { name: 'Extreme Cases', emoji: '⚖️', color: 'mlorange' },
    toolbox: { name: 'The Toolbox (6 Methods)', emoji: '🛠️', color: 'mlpurple' },
    quiz1: { name: 'Quiz 1: Mechanisms', emoji: '❓', color: 'mlgreen' },
    problems: { name: 'The 6 Problems', emoji: '⚠️', color: 'mlred' },
    quiz2: { name: 'Quiz 2: Problems', emoji: '❓', color: 'mlgreen' },
    integration: { name: 'Integration', emoji: '🔗', color: 'mlblue' },
    quiz3: { name: 'Quiz 3: Tasks', emoji: '❓', color: 'mlgreen' },
    conclusion: { name: 'Conclusion', emoji: '🎯', color: 'mlpurple' },
    appendix: { name: 'Technical Appendix', emoji: '📚', color: 'mlgray' }
  };

  return (
    <div
      className="section-menu-overlay fixed inset-0 bg-black/60 z-50"
      onClick={onClose}
    >
      {/* Sidebar */}
      <div
        className="section-menu-sidebar fixed right-0 top-0 bottom-0 w-96 bg-white shadow-2xl overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-mlpurple text-white px-6 py-4 sticky top-0 z-10">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold">Sections</h2>
            <button
              onClick={onClose}
              className="text-white text-2xl hover:text-mllavender transition"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Section List */}
        <div className="p-4 space-y-2">
          {Object.entries(sections).map(([sectionName, slideIds], index) => {
            const info = sectionInfo[sectionName] || { name: sectionName, emoji: '📄', color: 'mlgray' };
            const isActive = sectionName === currentSection;
            const slideCount = slideIds.length;
            const slideRange = slideCount > 1 ? `${slideIds[0]}-${slideIds[slideIds.length - 1]}` : `${slideIds[0]}`;

            return (
              <button
                key={sectionName}
                onClick={() => onSelectSection(sectionName)}
                className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? `bg-${info.color} text-white shadow-lg`
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{info.emoji}</span>
                    <div>
                      <div className="font-bold">{info.name}</div>
                      <div className={`text-xs ${isActive ? 'text-white/80' : 'text-gray-500'}`}>
                        {slideCount} slide{slideCount > 1 ? 's' : ''} (#{slideRange})
                      </div>
                    </div>
                  </div>
                  <div className="font-mono text-sm opacity-75">
                    {index + 1}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Keyboard Shortcuts */}
        <div className="border-t border-gray-200 p-6 bg-gray-50">
          <h3 className="font-bold text-sm text-gray-700 mb-3">Keyboard Shortcuts</h3>
          <div className="space-y-2 text-xs text-gray-600">
            <div className="flex justify-between">
              <span>Navigate slides:</span>
              <kbd className="kbd">← →</kbd>
            </div>
            <div className="flex justify-between">
              <span>First / Last slide:</span>
              <kbd className="kbd">Home / End</kbd>
            </div>
            <div className="flex justify-between">
              <span>Thumbnail grid:</span>
              <kbd className="kbd">G</kbd>
            </div>
            <div className="flex justify-between">
              <span>Section menu:</span>
              <kbd className="kbd">S</kbd>
            </div>
            <div className="flex justify-between">
              <span>Jump to section:</span>
              <kbd className="kbd">1-9</kbd>
            </div>
            <div className="flex justify-between">
              <span>Close overlay:</span>
              <kbd className="kbd">Esc</kbd>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SectionMenu;
