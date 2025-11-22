import { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import extractedSlides from '../../data/extractedSlides.json';
import './SlideViewer.css';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const SlideViewer = () => {
  const { slides, sections } = extractedSlides;
  const [currentSlide, setCurrentSlide] = useState(0);
  const slide = slides[currentSlide];

  const handleNext = () => {
    if (currentSlide < slides.length - 1) {
      setCurrentSlide(currentSlide + 1);
    }
  };

  const handlePrev = () => {
    if (currentSlide > 0) {
      setCurrentSlide(currentSlide - 1);
    }
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'ArrowRight') handleNext();
      if (e.key === 'ArrowLeft') handlePrev();
      if (e.key === 'Home') setCurrentSlide(0);
      if (e.key === 'End') setCurrentSlide(slides.length - 1);
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentSlide, slides.length]);

  // Get section name for current slide
  const getCurrentSection = () => {
    for (const [sectionName, slideIds] of Object.entries(sections)) {
      if (slideIds.includes(slide.id)) {
        return sectionName;
      }
    }
    return '';
  };

  const renderSlide = () => {
    return (
      <div className="slide-frame">
        {/* Title */}
        {slide.title && (
          <h2 className="slide-title">{slide.title}</h2>
        )}

        {/* Main content */}
        <div className="slide-content">
          {/* Figures (PDFs) */}
          {slide.figures && slide.figures.length > 0 && (
            <div className="figures-container">
              {slide.figures.map((fig, idx) => (
                <div
                  key={idx}
                  className="figure-wrapper"
                  style={{ width: `${fig.width * 100}%` }}
                >
                  <Document
                    file={`/figures/${fig.path}`}
                    className="pdf-document"
                  >
                    <Page pageNumber={1} width={800 * fig.width} />
                  </Document>
                </div>
              ))}
            </div>
          )}

          {/* Formulas */}
          {slide.formulas && slide.formulas.length > 0 && (
            <div className="formulas-container">
              {slide.formulas.map((formula, idx) => (
                <div key={idx} className="formula">
                  <BlockMath>{formula.content}</BlockMath>
                </div>
              ))}
            </div>
          )}

          {/* Lists */}
          {slide.lists && slide.lists.length > 0 && (
            <div className="lists-container">
              {slide.lists.map((list, idx) => (
                list.type === 'itemize' ? (
                  <ul key={idx} className="slide-list">
                    {list.items.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <ol key={idx} className="slide-list">
                    {list.items.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ol>
                )
              ))}
            </div>
          )}

          {/* Two-column content */}
          {slide.hasColumns && slide.columns && (
            <div className="two-columns">
              {slide.columns.map((col, idx) => (
                <div key={idx} className="column" style={{ flex: col.width }}>
                  {col.text && <p>{col.text}</p>}
                  {col.lists && col.lists.map((list, i) => (
                    list.type === 'itemize' ? (
                      <ul key={i}>
                        {list.items.map((item, j) => <li key={j}>{item}</li>)}
                      </ul>
                    ) : (
                      <ol key={i}>
                        {list.items.map((item, j) => <li key={j}>{item}</li>)}
                      </ol>
                    )
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Plain text */}
          {slide.text && !slide.hasColumns && slide.figures.length === 0 && (
            <div className="text-content">
              <p>{slide.text}</p>
            </div>
          )}
        </div>

        {/* Bottom note */}
        {slide.bottomNote && (
          <div className="bottom-note">
            {slide.bottomNote}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="slide-viewer">
      {/* Progress Bar */}
      <div className="progress-bar-container">
        <div
          className="progress-bar"
          style={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
        />
      </div>

      {/* Section Navigation */}
      <div className="section-nav">
        {Object.entries(sections).map(([sectionName, slideIds]) => (
          <button
            key={sectionName}
            onClick={() => setCurrentSlide(slideIds[0] - 1)}
            className={`section-button ${getCurrentSection() === sectionName ? 'active' : ''}`}
          >
            {sectionName.charAt(0).toUpperCase() + sectionName.slice(1)}
          </button>
        ))}
      </div>

      {/* Slide Container */}
      <div className="slide-container">
        {renderSlide()}
      </div>

      {/* Navigation Controls */}
      <div className="nav-controls">
        <button
          onClick={handlePrev}
          disabled={currentSlide === 0}
          className="nav-button"
        >
          ← Previous
        </button>

        <div className="slide-counter">
          Slide {currentSlide + 1} / {slides.length}
          <span className="section-label">({getCurrentSection()})</span>
        </div>

        <button
          onClick={handleNext}
          disabled={currentSlide === slides.length - 1}
          className="nav-button"
        >
          Next →
        </button>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="keyboard-hint">
        Use arrow keys (← →) or Home/End to navigate
      </div>
    </div>
  );
};

export default SlideViewer;
