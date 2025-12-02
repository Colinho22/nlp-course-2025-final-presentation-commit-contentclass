/**
 * Slide 5: Transition - What If We Explored More?
 * Complex vertical layout with:
 * - Green text (Greedy's choice)
 * - Tabular table (alternatives)
 * - Lavender colorbox (question)
 * - Pause reveal
 * - Orange text (answer)
 */

import { useState } from 'react';
import BottomNote from '../common/BottomNote';

const Slide5Transition = () => {
  const [revealed, setRevealed] = useState(false);

  return (
    <div className="slide-frame h-full flex flex-col p-10" onClick={() => !revealed && setRevealed(true)}>
      {/* Slide Title */}
      <h2 className="slide-title bg-gradient-to-r from-mllavender3 to-mllavender4 text-mlpurple px-6 py-3 -mx-10 -mt-10 mb-6 border-b-3 border-mlpurple font-bold text-2xl">
        Transition: What If We Explored More?
      </h2>

      {/* Main Content */}
      <div className="slide-content flex-1 flex flex-col justify-center space-y-6">
        {/* Greedy's choice */}
        <div className="text-center">
          <p className="text-lg">
            <strong>Greedy chose:</strong>{' '}
            <span className="text-mlgreen font-bold text-xl">"it" (P=0.50)</span>
          </p>
        </div>

        <div className="h-3" />

        {/* But it ignored */}
        <div className="text-center">
          <p className="text-lg font-bold mb-3">But it ignored these alternatives:</p>

          {/* Table of alternatives */}
          <table className="mx-auto border-collapse">
            <tbody>
              <tr>
                <td className="px-6 py-2 text-left font-mono">"a mouse"</td>
                <td className="px-6 py-2 text-right font-mono">P = 0.30 × 0.25 = 0.075</td>
              </tr>
              <tr>
                <td className="px-6 py-2 text-left font-mono">"the mouse"</td>
                <td className="px-6 py-2 text-right font-mono">P = 0.28 × 0.22 = 0.062</td>
              </tr>
              <tr>
                <td className="px-6 py-2 text-left font-mono">"something"</td>
                <td className="px-6 py-2 text-right font-mono">P = 0.15 × 0.18 = 0.027</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="h-5" />

        {/* Question colorbox */}
        <div className="flex justify-center">
          <div className="bg-mllavender4 rounded-lg px-12 py-6 w-[85%] text-center border-2 border-mllavender2">
            <p className="text-xl font-bold text-mlpurple">
              But what if we chose "a mouse" instead?
            </p>
          </div>
        </div>

        <div className="h-3" />

        {/* Answer reveal (after pause) */}
        {revealed && (
          <div className="text-center fade-in">
            <p className="text-mlorange font-bold text-xl">
              Answer: Better story continuation!<br />
              "The cat saw a mouse and the hunt began."
            </p>
          </div>
        )}

        {/* Click hint */}
        {!revealed && (
          <div className="text-center text-sm text-gray-400 animate-pulse">
            Click to reveal answer...
          </div>
        )}
      </div>

      {/* Bottom Note */}
      <BottomNote>
        From 1 path (greedy) to ALL paths (full search) - what could go wrong?
      </BottomNote>
    </div>
  );
};

export default Slide5Transition;
