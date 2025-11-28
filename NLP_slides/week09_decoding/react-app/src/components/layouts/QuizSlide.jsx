/**
 * QuizSlide Layout
 * Pattern C - Quiz with asymmetric columns + pause reveal (5% of slides)
 * Structure: Title + Two Columns (0.45 + 0.55) + Pause + Answer Colorbox + BottomNote
 * Used by: Slides 28, 36, 40 (Checkpoint Quizzes 1-3)
 */

import { useState } from 'react';
import BottomNote from '../common/BottomNote';
import { MarkdownText } from '../../utils/markdownParser.jsx';

const QuizSlide = ({ slide }) => {
  const { title, columns, bottomNote, metadata } = slide;
  const [revealed, setRevealed] = useState(false);

  // Extract answer from sections (appears after \pause)
  const answerSection = slide.sections?.find(s => s.type === 'colorbox' && s.pauseBefore);
  const answerColor = answerSection?.color?.includes('green') ? 'bg-mlgreen/30' :
                      answerSection?.color?.includes('blue') ? 'bg-mlblue/20' :
                      'bg-mllavender3';

  const handleClick = (e) => {
    if (e.target.tagName === 'BUTTON') return;
    if (metadata?.hasPause && !revealed) {
      setRevealed(true);
    }
  };

  return (
    <div className="slide-frame h-full flex flex-col p-10" onClick={handleClick}>
      {/* Slide Title */}
      {title && (
        <h2 className="slide-title bg-gradient-to-r from-mllavender3 to-mllavender4 text-mlpurple px-6 py-3 -mx-10 -mt-10 mb-6 border-b-3 border-mlpurple font-bold text-2xl">
          {title}
        </h2>
      )}

      {/* Quiz Columns - Asymmetric (0.45 + 0.55) */}
      <div className="slide-content flex-1">
        <div className="flex gap-8">
          {/* Left Column - Questions (0.45\textwidth) */}
          <div className="column w-[45%]">
            {columns && columns[0] && (
              <>
                <h3 className="font-bold text-lg text-mlpurple mb-3">
                  {columns[0].sections?.find(s => s.type === 'text')?.content?.includes('Methods') ? 'Methods:' : 'Questions:'}
                </h3>
                {columns[0].sections?.filter(s => s.type === 'list').map((list, i) => (
                  list.listType === 'enumerate' ? (
                    <ol key={i} className="list-decimal ml-6 space-y-2 text-gray-800">
                      {list.items.map((item, j) => (
                        <li key={j} className="marker:text-mlpurple marker:font-bold">
                          <span dangerouslySetInnerHTML={{ __html: item }} />
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <ul key={i} className="list-disc ml-6 space-y-2 text-gray-800">
                      {list.items.map((item, j) => (
                        <li key={j} className="marker:text-mlpurple">
                          <span dangerouslySetInnerHTML={{ __html: item }} />
                        </li>
                      ))}
                    </ul>
                  )
                ))}
              </>
            )}
          </div>

          {/* Right Column - Options (0.55\textwidth) */}
          <div className="column w-[55%]">
            {columns && columns[1] && (
              <>
                <h3 className="font-bold text-lg text-mlpurple mb-3">
                  Match to:
                </h3>
                {columns[1].sections?.filter(s => s.type === 'list').map((list, i) => (
                  <ul key={i} className="list-none space-y-2 text-gray-800">
                    {list.items.map((item, j) => (
                      <li key={j} className="pl-0">
                        <span dangerouslySetInnerHTML={{ __html: item }} />
                      </li>
                    ))}
                  </ul>
                ))}
              </>
            )}
          </div>
        </div>

        {/* Spacer - vspace{5mm} */}
        <div className="h-5" />

        {/* Answer Reveal - appears after \pause */}
        {revealed && answerSection && (
          <div className="flex justify-center fade-in">
            <div className={`${answerColor} px-8 py-4 rounded-lg w-[85%] text-center border border-mllavender2`}>
              <p className="font-bold text-gray-800">
                <MarkdownText>{answerSection.content}</MarkdownText>
              </p>
            </div>
          </div>
        )}

        {/* Click hint if not revealed */}
        {!revealed && metadata?.hasPause && (
          <div className="text-center text-sm text-gray-400 animate-pulse mt-4">
            Click to reveal answers...
          </div>
        )}
      </div>

      {/* Bottom Note */}
      <BottomNote>{bottomNote}</BottomNote>
    </div>
  );
};

export default QuizSlide;
