/**
 * Math Display Components
 * Wraps KaTeX with Beamer-specific macros
 */

import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

// Beamer math macros (from template_beamer_final.tex)
const katexMacros = {
  "\\given": "\\mid",  // Conditional probability
  "\\argmax": "\\operatorname*{argmax}",
  "\\argmin": "\\operatorname*{argmin}",
  "\\softmax": "\\operatorname{softmax}",
};

export const DisplayMath = ({ math, className = "" }) => {
  if (!math) return null;

  return (
    <div className={`math-display my-4 ${className}`}>
      <BlockMath math={math} macros={katexMacros} />
    </div>
  );
};

export const InlineMathComponent = ({ math }) => {
  if (!math) return null;

  return <InlineMath math={math} macros={katexMacros} />;
};

export const FormulaBlock = ({ formula, className = "" }) => {
  return (
    <div className={`formula-block bg-lightgray px-6 py-5 rounded-lg border-l-4 border-mlpurple my-4 ${className}`}>
      <BlockMath math={formula} macros={katexMacros} />
    </div>
  );
};

export default { DisplayMath, InlineMathComponent, FormulaBlock };
