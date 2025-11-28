/**
 * SingleColumnFigure Layout
 * Pattern A - Most common layout (45% of slides)
 * Structure: Title + Figure + Optional Text/Lists + BottomNote
 * Spacing: -0.3cm vspace after title, 2-3mm after figure
 */

import FigureDisplay from '../common/FigureDisplay';
import BottomNote from '../common/BottomNote';
import { DisplayMath } from '../common/MathDisplay';
import { MarkdownText } from '../../utils/markdownParser.jsx';

const SingleColumnFigure = ({ slide }) => {
  const { title, sections, bottomNote } = slide;

  // Extract different content types from sections
  const contentSections = sections || [];

  const renderSection = (section, index) => {
    switch (section.type) {
      case 'figure':
        return (
          <FigureDisplay
            key={`fig-${index}`}
            src={section.path}
            width={section.width}
            centered={true}
          />
        );

      case 'formula':
        return <DisplayMath key={`formula-${index}`} math={section.content} />;

      case 'list':
        return section.listType === 'itemize' ? (
          <ul key={`list-${index}`} className="slide-list list-disc ml-8 space-y-2 text-gray-800 text-lg">
            {section.items.map((item, j) => (
              <li key={j} className="marker:text-mlpurple marker:text-xl">
                <MarkdownText>{item}</MarkdownText>
              </li>
            ))}
          </ul>
        ) : (
          <ol key={`list-${index}`} className="slide-list list-decimal ml-8 space-y-2 text-gray-800 text-lg">
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
          <div key={`text-${index}`} className="text-center text-gray-800 text-lg leading-relaxed mb-3">
            <p><MarkdownText>{section.content}</MarkdownText></p>
          </div>
        );

      case 'vspace':
        // Convert LaTeX spacing
        const isNegative = section.value.includes('-');
        const value = section.value.replace('-', '');
        const pixels = value.includes('cm') ? parseFloat(value) * 37.8 :
                       value.includes('mm') ? parseFloat(value) * 3.78 : 8;
        return <div key={`space-${index}`} style={{ height: isNegative ? 0 : `${pixels}px`, marginTop: isNegative ? `-${pixels}px` : 0 }} />;

      case 'pause':
        return null; // Handle in parent

      default:
        return null;
    }
  };

  return (
    <div className="slide-frame h-full flex flex-col p-10">
      {/* Slide Title */}
      {title && (
        <h2 className="slide-title bg-gradient-to-r from-mllavender3 to-mllavender4 text-mlpurple px-6 py-3 -mx-10 -mt-10 mb-6 border-b-3 border-mlpurple font-bold text-2xl">
          {title}
        </h2>
      )}

      {/* Main Content - Render all sections in order */}
      <div className="slide-content flex-1">
        {contentSections.map((section, index) => renderSection(section, index))}
      </div>

      {/* Bottom Note */}
      <BottomNote>{bottomNote}</BottomNote>
    </div>
  );
};

export default SingleColumnFigure;
