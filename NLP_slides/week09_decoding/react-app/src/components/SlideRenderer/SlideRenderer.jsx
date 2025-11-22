/**
 * SlideRenderer - Pattern Matching Router
 * Routes each slide to the appropriate layout component based on slide.layout
 */

import TitleSlide from '../layouts/TitleSlide';
import SingleColumnFigure from '../layouts/SingleColumnFigure';
import TwoColumnEqual from '../layouts/TwoColumnEqual';
import QuizSlide from '../layouts/QuizSlide';
import FourFigureGrid from '../layouts/FourFigureGrid';
import Slide42Timeline from '../special/Slide42Timeline';
import Slide5Transition from '../special/Slide5Transition';

const SlideRenderer = ({ slide, onNextSlide }) => {
  if (!slide) {
    return (
      <div className="slide-frame h-full flex items-center justify-center">
        <p className="text-gray-500">Slide not found</p>
      </div>
    );
  }

  // Special case slides (by ID)
  if (slide.id === 5) {
    return <Slide5Transition />;
  }

  if (slide.id === 42) {
    return <Slide42Timeline />;
  }

  // Route to correct layout based on pattern
  switch (slide.layout) {
    case 'title-slide':
      // Slides 1 and 43
      return slide.id === 1 ? (
        <TitleSlide
          title="Decoding Strategies"
          subtitle="Week 9: From Probabilities to Text"
          date="November 2025"
        />
      ) : (
        <TitleSlide
          title="Technical Appendix"
          subtitle="25 slides: Complete mathematical treatment"
          appendixSections={[
            { range: 'A1-A5', title: 'Beam Search: Complete Mathematics' },
            { range: 'A6-A10', title: 'Sampling: Formal Foundations' },
            { range: 'A11-A14', title: 'Contrastive Search & Degeneration' },
            { range: 'A15-A19', title: 'Advanced Topics & Production' },
            { range: 'A20-A25', title: 'The 6 Problems - Technical Analysis (NEW)' }
          ]}
        />
      );

    case 'single-column-figure':
      // Most common: 45% of slides
      return <SingleColumnFigure slide={slide} />;

    case 'two-column-equal':
    case 'two-column-asymmetric':
    case 'two-column-custom':
      // 19% of slides
      return <TwoColumnEqual slide={slide} />;

    case 'quiz':
      // 3 checkpoint quizzes
      return <QuizSlide slide={slide} onNextSlide={onNextSlide} />;

    case 'four-figure-grid':
      // Slide 32 only
      return <FourFigureGrid slide={slide} />;

    case 'tikz-diagram':
      // Slide 42 - Timeline SVG
      return <Slide42Timeline />;

    case 'single-column-text':
      // Text-only slides (no figures)
      return <SingleColumnFigure slide={slide} />;

    case 'appendix-divider':
      // Same as title slide
      return <TitleSlide {...slide} />;

    default:
      // Fallback: use SingleColumnFigure as generic renderer
      console.warn(`Unknown layout: ${slide.layout} for slide ${slide.id}`);
      return <SingleColumnFigure slide={slide} />;
  }
};

export default SlideRenderer;
