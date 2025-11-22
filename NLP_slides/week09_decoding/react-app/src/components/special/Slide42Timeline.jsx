/**
 * Slide 42: Complete Journey Timeline
 * Converts TikZ diagram to React SVG
 * Shows: Probabilities → Decoding → Text → Application
 * Arrow with 4 milestone circles, gradient colors
 */

import BottomNote from '../common/BottomNote';

const TimelineSVG = () => {
  return (
    <svg viewBox="0 0 1000 180" className="w-[90%] mx-auto">
      {/* Definitions */}
      <defs>
        {/* Arrow marker */}
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L9,3 z" fill="#3333B2" />
        </marker>

        {/* Gradient for last circle */}
        <linearGradient id="purpleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: '#3333B2', stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: '#ADADE0', stopOpacity: 1 }} />
        </linearGradient>
      </defs>

      {/* Main arrow */}
      <line
        x1="80"
        y1="100"
        x2="920"
        y2="100"
        stroke="#3333B2"
        strokeWidth="5"
        markerEnd="url(#arrowhead)"
      />

      {/* Milestone 1: Probabilities (start - lavender) */}
      <circle cx="120" cy="100" r="30" fill="#ADADE0" stroke="#3333B2" strokeWidth="3" />
      <text
        x="120"
        y="55"
        textAnchor="middle"
        className="fill-mlpurple font-bold text-sm"
      >
        Probabilities
      </text>
      <text
        x="120"
        y="70"
        textAnchor="middle"
        className="fill-gray-600 text-xs"
      >
        (Model Output)
      </text>

      {/* Milestone 2: Decoding (lavender) */}
      <circle cx="360" cy="100" r="30" fill="#ADADE0" stroke="#3333B2" strokeWidth="3" />
      <text
        x="360"
        y="55"
        textAnchor="middle"
        className="fill-mlpurple font-bold text-sm"
      >
        Decoding
      </text>
      <text
        x="360"
        y="70"
        textAnchor="middle"
        className="fill-gray-600 text-xs"
      >
        (6 Methods)
      </text>

      {/* Milestone 3: Text (lavender) */}
      <circle cx="600" cy="100" r="30" fill="#ADADE0" stroke="#3333B2" strokeWidth="3" />
      <text
        x="600"
        y="55"
        textAnchor="middle"
        className="fill-mlpurple font-bold text-sm"
      >
        Text
      </text>
      <text
        x="600"
        y="70"
        textAnchor="middle"
        className="fill-gray-600 text-xs"
      >
        (Human-Readable)
      </text>

      {/* Milestone 4: Application (end - purple, larger) */}
      <circle cx="880" cy="100" r="35" fill="url(#purpleGradient)" stroke="#3333B2" strokeWidth="4" />
      <text
        x="880"
        y="55"
        textAnchor="middle"
        className="fill-white font-bold text-base"
      >
        Application
      </text>
      <text
        x="880"
        y="72"
        textAnchor="middle"
        className="fill-white text-xs"
      >
        (Real-World Use)
      </text>
    </svg>
  );
};

const Slide42Timeline = () => {
  return (
    <div className="slide-frame h-full flex flex-col p-10">
      {/* Slide Title */}
      <h2 className="slide-title bg-gradient-to-r from-mllavender3 to-mllavender4 text-mlpurple px-6 py-3 -mx-10 -mt-10 mb-6 border-b-3 border-mlpurple font-bold text-2xl">
        From Probabilities to Text: The Complete Journey
      </h2>

      {/* Timeline */}
      <div className="flex-1 flex flex-col justify-center">
        <TimelineSVG />

        {/* vspace */}
        <div className="h-10" />

        {/* What We Learned list */}
        <div>
          <p className="font-bold text-mlpurple text-lg mb-3">What We Learned:</p>
          <ul className="list-disc ml-8 space-y-2 text-gray-800">
            <li className="marker:text-mlpurple">
              <strong>The Extremes</strong>: Greedy (too narrow) vs Full Search (too broad)
            </li>
            <li className="marker:text-mlpurple">
              <strong>6 Methods</strong>: Each balances quality and diversity differently
            </li>
            <li className="marker:text-mlpurple">
              <strong>6 Problems</strong>: Repetition, diversity, balance, missing paths, tail junk, degeneration
            </li>
            <li className="marker:text-mlpurple">
              <strong>Task Matters</strong>: Factual → Greedy, Creative → Nucleus, Long → Contrastive
            </li>
            <li className="marker:text-mlpurple">
              <strong>Modern Standard</strong>: Nucleus (p=0.9) + Temperature (T=0.7) most common
            </li>
          </ul>
        </div>
      </div>

      {/* Bottom Note */}
      <BottomNote>
        Complete pipeline from model training to text generation - decoding is the critical final step
      </BottomNote>
    </div>
  );
};

export default Slide42Timeline;
