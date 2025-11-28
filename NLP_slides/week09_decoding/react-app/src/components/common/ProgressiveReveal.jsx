/**
 * Progressive Reveal Component
 * Implements Beamer \pause functionality
 * Click anywhere on slide to reveal next section
 */

import { useState } from 'react';

const ProgressiveReveal = ({ children, totalSteps, onAllRevealed }) => {
  const [currentStep, setCurrentStep] = useState(0);

  const handleClick = (e) => {
    // Don't intercept clicks on interactive elements
    if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') {
      return;
    }

    e.stopPropagation();

    if (currentStep < totalSteps) {
      setCurrentStep(s => s + 1);
    } else if (onAllRevealed) {
      onAllRevealed();
    }
  };

  return (
    <div onClick={handleClick} className="progressive-reveal h-full cursor-pointer">
      {/* Map children with visibility based on step */}
      {Array.isArray(children) ? (
        children.map((child, index) => (
          index <= currentStep ? (
            <div key={index} className="fade-in" style={{ animationDelay: '0s' }}>
              {child}
            </div>
          ) : null
        ))
      ) : (
        <div className={currentStep > 0 ? 'fade-in' : ''}>{children}</div>
      )}

      {/* Visual indicator that more content is hidden */}
      {currentStep < totalSteps && (
        <div className="text-center mt-4 text-xs text-gray-400 animate-pulse">
          Click to reveal more...
        </div>
      )}
    </div>
  );
};

export default ProgressiveReveal;
