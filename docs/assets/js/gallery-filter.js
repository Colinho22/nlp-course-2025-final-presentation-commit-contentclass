// Gallery Filter and Lightbox functionality
document.addEventListener('DOMContentLoaded', function() {
  // Filter functionality
  const filterButtons = document.querySelectorAll('.filter-btn');
  const chartCards = document.querySelectorAll('.chart-card');

  filterButtons.forEach(function(button) {
    button.addEventListener('click', function() {
      const filter = this.dataset.filter;

      // Update active button
      filterButtons.forEach(btn => btn.classList.remove('active'));
      this.classList.add('active');

      // Filter cards
      chartCards.forEach(function(card) {
        if (filter === 'all' || card.dataset.category === filter) {
          card.classList.remove('hidden');
        } else {
          card.classList.add('hidden');
        }
      });
    });
  });

  // Lightbox functionality
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightbox-img');
  const lightboxCaption = document.getElementById('lightbox-caption');
  const closeBtn = document.querySelector('.lightbox-close');
  const prevBtn = document.querySelector('.lightbox-nav.prev');
  const nextBtn = document.querySelector('.lightbox-nav.next');

  let currentIndex = 0;
  let visibleCards = [];

  window.openLightbox = function(src, caption) {
    // Get visible cards
    visibleCards = Array.from(chartCards).filter(card => !card.classList.contains('hidden'));
    currentIndex = visibleCards.findIndex(card => card.querySelector('img').src.includes(src.split('/').pop()));

    lightboxImg.src = src;
    lightboxCaption.textContent = caption;
    lightbox.classList.add('active');
    document.body.style.overflow = 'hidden';
  };

  function closeLightbox() {
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
  }

  function showPrev() {
    if (visibleCards.length === 0) return;
    currentIndex = (currentIndex - 1 + visibleCards.length) % visibleCards.length;
    const card = visibleCards[currentIndex];
    lightboxImg.src = card.querySelector('img').src;
    lightboxCaption.textContent = card.querySelector('.chart-title').textContent;
  }

  function showNext() {
    if (visibleCards.length === 0) return;
    currentIndex = (currentIndex + 1) % visibleCards.length;
    const card = visibleCards[currentIndex];
    lightboxImg.src = card.querySelector('img').src;
    lightboxCaption.textContent = card.querySelector('.chart-title').textContent;
  }

  closeBtn?.addEventListener('click', closeLightbox);
  prevBtn?.addEventListener('click', showPrev);
  nextBtn?.addEventListener('click', showNext);

  // Close on background click
  lightbox?.addEventListener('click', function(e) {
    if (e.target === lightbox) {
      closeLightbox();
    }
  });

  // Keyboard navigation
  document.addEventListener('keydown', function(e) {
    if (!lightbox?.classList.contains('active')) return;

    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowLeft') showPrev();
    if (e.key === 'ArrowRight') showNext();
  });
});
