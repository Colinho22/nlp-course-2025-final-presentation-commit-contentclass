import './Madrid.css';

const MadridHeader = ({ currentSection, sections, onSectionClick }) => {
  return (
    <header className="madrid-header bg-mllavender3 border-b-2 border-mlpurple px-6 py-2">
      <div className="flex justify-between items-center">
        <div className="text-sm font-semibold text-mlpurple">
          Week 9: Decoding Strategies
        </div>

        {/* Navigation dots (Madrid theme style) */}
        <nav className="flex gap-1.5" aria-label="Section navigation">
          {Object.entries(sections).map(([sectionName, slideIds], index) => (
            <button
              key={sectionName}
              onClick={() => onSectionClick(slideIds[0] - 1)}
              className={`w-2.5 h-2.5 rounded-full transition-all hover:scale-125 ${
                sectionName === currentSection
                  ? 'bg-mlpurple ring-2 ring-mlpurple ring-offset-1'
                  : 'bg-mllavender hover:bg-mllavender2'
              }`}
              aria-label={`Jump to ${sectionName} section`}
              title={sectionName.charAt(0).toUpperCase() + sectionName.slice(1)}
            />
          ))}
        </nav>
      </div>
    </header>
  );
};

export default MadridHeader;
