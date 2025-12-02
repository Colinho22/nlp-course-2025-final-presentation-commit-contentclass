/**
 * Main App - Sidebar Learning Interface
 * 280px sidebar + flexible main area with progress tracking
 */

import { useState, useEffect } from 'react';
import { ThemeProvider, CssBaseline, Box } from '@mui/material';
import Sidebar from './components/Sidebar';
import ProgressBar from './components/ProgressBar';
import SlideRenderer from './components/SlideRenderer';
import Navigation from './components/Navigation';
import useProgress from './hooks/useProgress';
import theme from './theme';
import slideData from './week09_slides_complete.json';

function App() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const { completedSlides, markSlideComplete } = useProgress();
  const { slides } = slideData;

  // Auto-mark slide as viewed after 3 seconds
  useEffect(() => {
    const timer = setTimeout(() => {
      markSlideComplete(slides[currentSlide].id);
    }, 3000);
    return () => clearTimeout(timer);
  }, [currentSlide]);

  // Keyboard navigation
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' && currentSlide < slides.length - 1) {
        setCurrentSlide(s => s + 1);
      } else if (e.key === 'ArrowLeft' && currentSlide > 0) {
        setCurrentSlide(s => s - 1);
      } else if (e.key === 'Home') {
        setCurrentSlide(0);
      } else if (e.key === 'End') {
        setCurrentSlide(slides.length - 1);
      }
    };

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [currentSlide, slides.length]);

  const handleGoalClick = (slideIndex) => {
    setCurrentSlide(slideIndex);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* Sidebar - 280px fixed */}
        <Sidebar
          currentSlide={currentSlide}
          onGoalClick={handleGoalClick}
          completedSlides={completedSlides}
        />

        {/* Main Content Area */}
        <Box sx={{ flexGrow: 1, ml: '280px', display: 'flex', flexDirection: 'column' }}>
          {/* Top Progress Bar */}
          <ProgressBar current={currentSlide} total={slides.length} />

          {/* Slide Content - Fixed width, centered */}
          <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', p: 4, overflow: 'auto' }}>
            <Box sx={{ width: '100%', maxWidth: '1200px' }}>
              <SlideRenderer slide={slides[currentSlide]} />
            </Box>
          </Box>

          {/* Bottom Navigation */}
          <Navigation
            onPrev={() => setCurrentSlide(s => s - 1)}
            onNext={() => setCurrentSlide(s => s + 1)}
            onFirst={() => setCurrentSlide(0)}
            onLast={() => setCurrentSlide(slides.length - 1)}
            canPrev={currentSlide > 0}
            canNext={currentSlide < slides.length - 1}
          />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
