/**
 * Main Presentation Container
 * Fixed 16:9 aspect ratio with Madrid theme
 * Full navigation controls: keyboard + buttons + thumbnails + sections
 */

import { useState, useEffect } from 'react';
import MadridHeader from '../Madrid/MadridHeader';
import MadridFooter from '../Madrid/MadridFooter';
import SlideRenderer from '../SlideRenderer/SlideRenderer';
import ThumbnailGrid from './ThumbnailGrid';
import SectionMenu from './SectionMenu';
import Controls from './Controls';
import slideData from '../../data/week09_slides_complete.json';
import './Presentation.css';

const Presentation = () => {
  const { slides, metadata } = slideData;
  const sections = metadata.sections;

  const [currentSlide, setCurrentSlide] = useState(0);
  const [showThumbnails, setShowThumbnails] = useState(false);
  const [showSectionMenu, setShowSectionMenu] = useState(false);

  const slide = slides[currentSlide];

  // Navigation functions
  const nextSlide = () => {
    if (currentSlide < slides.length - 1) {
      setCurrentSlide(currentSlide + 1);
    }
  };

  const prevSlide = () => {
    if (currentSlide > 0) {
      setCurrentSlide(currentSlide - 1);
    }
  };

  const jumpToSlide = (index) => {
    if (index >= 0 && index < slides.length) {
      setCurrentSlide(index);
      setShowThumbnails(false);
      setShowSectionMenu(false);
    }
  };

  const jumpToSection = (sectionName) => {
    const sectionSlides = Object.entries(sections).find(([name]) => name === sectionName)?.[1];
    if (sectionSlides && sectionSlides.length > 0) {
      jumpToSlide(sectionSlides[0] - 1);  // Convert to 0-indexed
    }
  };

  // Get current section
  const getCurrentSection = () => {
    for (const [sectionName, slideIds] of Object.entries(sections)) {
      if (slideIds.includes(slide.id)) {
        return sectionName;
      }
    }
    return '';
  };

  // Comprehensive keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't interfere with input fields
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      switch (e.key) {
        case 'ArrowRight':
        case ' ': // Space bar
        case 'PageDown':
          e.preventDefault();
          nextSlide();
          break;

        case 'ArrowLeft':
        case 'PageUp':
          e.preventDefault();
          prevSlide();
          break;

        case 'Home':
          e.preventDefault();
          jumpToSlide(0);
          break;

        case 'End':
          e.preventDefault();
          jumpToSlide(slides.length - 1);
          break;

        case 'Escape':
          e.preventDefault();
          setShowThumbnails(false);
          setShowSectionMenu(false);
          break;

        case 'g':
        case 'G':
          e.preventDefault();
          setShowThumbnails(s => !s);
          break;

        case 's':
        case 'S':
          e.preventDefault();
          setShowSectionMenu(s => !s);
          break;

        default:
          // Number keys 1-9 for quick section jump
          if (e.key >= '1' && e.key <= '9') {
            const sectionIndex = parseInt(e.key) - 1;
            const sectionNames = Object.keys(sections);
            if (sectionIndex < sectionNames.length) {
              jumpToSection(sectionNames[sectionIndex]);
            }
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSlide, slides.length]);

  return (
    <div className="presentation-container h-screen flex flex-col bg-gray-900">
      {/* Madrid Header */}
      <MadridHeader
        currentSection={getCurrentSection()}
        sections={sections}
        onSectionClick={jumpToSlide}
      />

      {/* Main Slide Viewport - Fixed 16:9 */}
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="slide-viewport aspect-beamer bg-white rounded-lg shadow-2xl overflow-hidden max-w-full max-h-full">
          <SlideRenderer
            slide={slide}
            onNextSlide={nextSlide}
          />
        </div>
      </main>

      {/* Madrid Footer */}
      <MadridFooter
        current={currentSlide + 1}
        total={slides.length}
        title="Week 9: Decoding Strategies"
        author="NLP Course 2025"
      />

      {/* Full Controls */}
      <Controls
        onPrev={prevSlide}
        onNext={nextSlide}
        onToggleThumbnails={() => setShowThumbnails(s => !s)}
        onToggleSections={() => setShowSectionMenu(s => !s)}
        current={currentSlide + 1}
        total={slides.length}
        currentSection={getCurrentSection()}
        canPrev={currentSlide > 0}
        canNext={currentSlide < slides.length - 1}
      />

      {/* Thumbnail Grid Overlay */}
      {showThumbnails && (
        <ThumbnailGrid
          slides={slides}
          currentSlide={currentSlide}
          onSelect={jumpToSlide}
          onClose={() => setShowThumbnails(false)}
        />
      )}

      {/* Section Menu Sidebar */}
      {showSectionMenu && (
        <SectionMenu
          sections={sections}
          currentSlide={currentSlide}
          currentSection={getCurrentSection()}
          onSelectSection={jumpToSection}
          onClose={() => setShowSectionMenu(false)}
        />
      )}
    </div>
  );
};

export default Presentation;
