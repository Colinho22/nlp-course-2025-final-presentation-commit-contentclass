// Simple client-side search
(function() {
  let searchIndex = null;
  let searchData = null;

  // Load search index
  async function loadSearchIndex() {
    try {
      const baseUrl = document.querySelector('meta[name="baseurl"]')?.content || '';
      const response = await fetch(baseUrl + '/search.json');
      searchData = await response.json();
      searchIndex = buildIndex(searchData);
    } catch (error) {
      console.error('Failed to load search index:', error);
    }
  }

  // Simple index builder
  function buildIndex(data) {
    return data.map((item, index) => ({
      ...item,
      index,
      searchText: (item.title + ' ' + item.content + ' ' + (item.categories || []).join(' ')).toLowerCase()
    }));
  }

  // Search function
  function search(query) {
    if (!searchIndex || !query) return [];

    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    if (terms.length === 0) return [];

    return searchIndex
      .map(item => {
        let score = 0;
        terms.forEach(term => {
          // Title match (weighted higher)
          if (item.title.toLowerCase().includes(term)) {
            score += 10;
          }
          // Content match
          if (item.searchText.includes(term)) {
            score += 1;
          }
        });
        return { ...item, score };
      })
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  }

  // Render results
  function renderResults(results, container) {
    if (results.length === 0) {
      container.innerHTML = '<div class="search-result-item"><p>No results found</p></div>';
      return;
    }

    container.innerHTML = results.map(item => `
      <a href="${item.url}" class="search-result-item">
        <div class="result-title">${escapeHtml(item.title)}</div>
        <div class="result-excerpt">${escapeHtml(truncate(item.content, 100))}</div>
      </a>
    `).join('');
  }

  // Utility functions
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function truncate(text, length) {
    if (!text) return '';
    return text.length > length ? text.substring(0, length) + '...' : text;
  }

  // Initialize search
  document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('.search-input');
    const searchResults = document.querySelector('.search-results');

    if (!searchInput || !searchResults) return;

    loadSearchIndex();

    // Debounce search
    let timeout;
    searchInput.addEventListener('input', function() {
      clearTimeout(timeout);
      const query = this.value.trim();

      if (query.length < 2) {
        searchResults.classList.remove('active');
        return;
      }

      timeout = setTimeout(function() {
        const results = search(query);
        renderResults(results, searchResults);
        searchResults.classList.add('active');
      }, 200);
    });

    // Close on outside click
    document.addEventListener('click', function(e) {
      if (!e.target.closest('.search-container')) {
        searchResults.classList.remove('active');
      }
    });

    // Close on escape
    searchInput.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        searchResults.classList.remove('active');
        this.blur();
      }
    });
  });
})();
