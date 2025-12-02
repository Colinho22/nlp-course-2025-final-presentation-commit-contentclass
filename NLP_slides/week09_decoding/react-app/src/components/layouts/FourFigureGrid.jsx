/**
 * FourFigureGrid Layout
 * Pattern E - 2x2 grid of figures (Slide 32 only)
 * Structure: Title + 2x2 Grid of Figures + BottomNote
 * Spacing: Tight (-2mm, -0.5cm vspace), 2mm between stacked figures
 */

import FigureDisplay from '../common/FigureDisplay';
import BottomNote from '../common/BottomNote';

const FourFigureGrid = ({ slide }) => {
  const { title, columns, bottomNote } = slide;

  // Extract figures from columns
  const leftFigures = columns?.[0]?.sections?.filter(s => s.type === 'figure') || [];
  const rightFigures = columns?.[1]?.sections?.filter(s => s.type === 'figure') || [];

  return (
    <div className="slide-frame h-full flex flex-col p-10">
      {/* Slide Title */}
      {title && (
        <h2 className="slide-title bg-gradient-to-r from-mllavender3 to-mllavender4 text-mlpurple px-6 py-3 -mx-10 -mt-10 mb-4 border-b-3 border-mlpurple font-bold text-2xl">
          {title}
        </h2>
      )}

      {/* Main Content - Tight spacing */}
      <div className="slide-content flex-1 -mt-4 -mb-2">
        {/* Two-column grid */}
        <div className="flex gap-8 h-full">
          {/* Left Column - 2 stacked figures */}
          <div className="column w-[48%] flex flex-col gap-2">
            {leftFigures.map((fig, i) => (
              <div key={i} className="flex-1">
                <FigureDisplay
                  src={fig.path}
                  width={1.0}  // Full column width
                  centered={false}
                />
              </div>
            ))}
          </div>

          {/* Right Column - 2 stacked figures */}
          <div className="column w-[48%] flex flex-col gap-2">
            {rightFigures.map((fig, i) => (
              <div key={i} className="flex-1">
                <FigureDisplay
                  src={fig.path}
                  width={1.0}  // Full column width
                  centered={false}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Note */}
      <BottomNote>{bottomNote}</BottomNote>
    </div>
  );
};

export default FourFigureGrid;
