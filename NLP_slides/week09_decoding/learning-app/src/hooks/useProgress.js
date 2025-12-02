/**
 * Progress Tracking Hook
 * Persists slide completion to localStorage
 */

import { useState, useEffect } from 'react';

const STORAGE_KEY = 'week9-decoding-progress';

export const useProgress = () => {
  const [completedSlides, setCompletedSlides] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  // Persist to localStorage whenever completedSlides changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(completedSlides));
  }, [completedSlides]);

  const markSlideComplete = (slideId) => {
    setCompletedSlides(prev => ({
      ...prev,
      [slideId]: true
    }));
  };

  const isSlideComplete = (slideId) => {
    return Boolean(completedSlides[slideId]);
  };

  const getCompletionPercentage = (slideIds) => {
    const completed = slideIds.filter(id => completedSlides[id]).length;
    return Math.round((completed / slideIds.length) * 100);
  };

  const resetProgress = () => {
    setCompletedSlides({});
    localStorage.removeItem(STORAGE_KEY);
  };

  return {
    completedSlides,
    markSlideComplete,
    isSlideComplete,
    getCompletionPercentage,
    resetProgress
  };
};

export default useProgress;
