/**
 * TwoColumnEqual Layout
 * Pattern B - Two-column layout with 0.48\textwidth each (19% of slides)
 * Structure: Title + Two Columns (equal width) + BottomNote
 * Used for: Algorithm comparisons, method details, side-by-side content
 */

import BottomNote from '../common/BottomNote';
import FigureDisplay from '../common/FigureDisplay';
import { DisplayMath } from '../common/MathDisplay';
import { MarkdownText } from '../../utils/markdownParser.jsx';

const TwoColumnEqual = ({ slide }) => {
  const { title, columns, bottomNote } = slide;

  const renderColumnContent = (column) => {
    if (!column.sections) return null;

    return column.sections.map((section, i) => {
      switch (section.type) {
        case 'figure':
          return (
            <FigureDisplay
              key={i}
              src={section.path}
              width={section.width}
              centered={true}
            />
          );

        case 'formula':
          return <DisplayMath key={i} math={section.content} />;

        case 'list':
          return section.listType === 'itemize' ? (
            <ul key={i} className="list-disc ml-6 space-y-1.5 text-gray-800">
              {section.items.map((item, j) => (
                <li key={j} className="marker:text-mlpurple">
                  <MarkdownText>{item}</MarkdownText>
                </li>
              ))}
            </ul>
          ) : (
            <ol key={i} className="list-decimal ml-6 space-y-1.5 text-gray-800">
              {section.items.map((item, j) => (
                <li key={j} className="marker:text-mlpurple marker:font-bold">
                  <MarkdownText>{item}</MarkdownText>
                </li>
              ))}
            </ol>
          );

        case 'text':
        case 'formattedText':
          return (
            <div key={i} className="text-gray-800 leading-relaxed mb-2">
              <p><MarkdownText>{section.content}</MarkdownText></p>
            </div>
          );

        case 'vspace':
          // Convert LaTeX spacing to Tailwind
          const spacing = section.value.includes('mm')
            ? parseInt(section.value) / 4  // rough mm to rem
            : section.value.includes('cm')
            ? parseInt(section.value.replace('-', '')) * 10 / 4
            : 2;
          return <div key={i} style={{ height: `${spacing}px` }} />;

        default:
          return null;
      }
    });
  };

  return (
    <div className="slide-frame h-full flex flex-col p-10">
      {/* Slide Title */}
      {title && (
        <h2 className="slide-title bg-gradient-to-r from-mllavender3 to-mllavender4 text-mlpurple px-6 py-3 -mx-10 -mt-10 mb-6 border-b-3 border-mlpurple font-bold text-2xl">
          {title}
        </h2>
      )}

      {/* Two Columns - [T] top-aligned */}
      <div className="slide-content flex-1 flex gap-10 items-start">
        {/* Left Column - 0.48\textwidth */}
        <div className="column w-[48%]">
          {columns && columns[0] && renderColumnContent(columns[0])}
        </div>

        {/* Right Column - 0.48\textwidth */}
        <div className="column w-[48%]">
          {columns && columns[1] && renderColumnContent(columns[1])}
        </div>
      </div>

      {/* Bottom Note */}
      <BottomNote>{bottomNote}</BottomNote>
    </div>
  );
};

export default TwoColumnEqual;
