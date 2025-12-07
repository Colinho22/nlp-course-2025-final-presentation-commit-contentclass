#!/usr/bin/env python3
"""
Moodle Course Site Generator
Generates Moodle-focused pages with schedule, PDFs, notebooks, and assignments
"""

import json
from pathlib import Path
from datetime import datetime

# Moodle course configuration
MOODLE_CONFIG = {
    'course_name': 'Text Analytics (dv-) HS25',
    'course_short': 'TA HS25',
    'github_url': 'https://github.com/Digital-AI-Finance/Natural-Language-Processing',
    'streamlit_app': 'https://nlp-evolution.streamlit.app/',
    'output_dir': Path(__file__).parent.parent.parent / 'docs' / 'moodle'  # Absolute path
}

# Course schedule (13 sessions from Semesterbeschreibung)
SCHEDULE = [
    {'date': '12.09.25', 'topic': 'Shakespeare und N-Grams', 'description': 'Unser eigenes Shakespeare Sonnet', 'mandatory': False},
    {'date': '19.09.25', 'topic': 'Word Embeddings', 'description': 'Einfuhrung in Word Embeddings', 'mandatory': False},
    {'date': '26.09.25', 'topic': 'Neuronale Netze', 'description': 'Function Approximation', 'mandatory': False},
    {'date': '03.10.25', 'topic': 'RNNs', 'description': 'Recurrent Neural Networks', 'mandatory': False},
    {'date': '17.10.25', 'topic': 'LSTMs', 'description': 'Long Short-Term Memory', 'mandatory': False},
    {'date': '24.10.25', 'topic': 'Zwischenprasentation', 'description': '10 min + 10 min Q&A', 'mandatory': True},
    {'date': '31.10.25', 'topic': 'Sequence-to-Sequence', 'description': 'Predicting the next sentence', 'mandatory': False},
    {'date': '07.11.25', 'topic': 'Transformers', 'description': 'The Transformer revolution', 'mandatory': False},
    {'date': '14.11.25', 'topic': 'Multi-Agent LLMs', 'description': 'Multi-Agent Systems', 'mandatory': False},
    {'date': '21.11.25', 'topic': 'Kurzprasentation Option 1', 'description': '5 min + 5 min Q&A', 'mandatory': True},
    {'date': '28.11.25', 'topic': 'Kurzprasentation Option 2', 'description': '5 min + 5 min Q&A', 'mandatory': True},
    {'date': '05.12.25', 'topic': 'Decoding Strategies', 'description': 'Text Generation Methods', 'mandatory': False},
    {'date': '12.12.25', 'topic': 'Abschlussprasentation', 'description': '15-20 min + 5-10 min Q&A', 'mandatory': True}
]

# Assessment weights
ASSIGNMENTS = [
    {'name': 'Kurzprasentation', 'weight': 20, 'type': 'individual', 'description': '5 min + 5 min Q&A, cover one course topic'},
    {'name': 'Zwischenprasentation', 'weight': 10, 'type': 'team', 'description': '10 min + 10 min Q&A, describe 4 NLP methods'},
    {'name': 'Abschlussprasentation', 'weight': 70, 'type': 'team', 'description': '15-20 min + Q&A, final project presentation'}
]

# Moodle amber accent theme with dark mode
MOODLE_STYLES = '''
:root {
  /* Light mode (default) */
  --moodle-accent: #f59e0b;
  --moodle-accent-light: #fef3c7;
  --moodle-accent-dark: #d97706;
  --secondary-accent: #3333B2;
  --secondary-accent-light: #E8E8F4;
  --text-dark: #1e293b;
  --text-gray: #4b5563;
  --text-muted: #64748b;
  --bg-light: #f8fafc;
  --bg-white: #ffffff;
  --bg-card: #ffffff;
  --bg-card-hover: #ffffff;
  --bg-input: #f6f8fa;
  --border: #e2e8f0;
  --border-light: #f0f0f0;
  --purple: #3333B2;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  --hero-gradient: linear-gradient(135deg, var(--moodle-accent-dark) 0%, var(--moodle-accent) 100%);
  --info-bg: #fef3c7;
  --info-border: #f59e0b;
  --info-text: #78350f;
  --info-strong: #92400e;
  --code-bg: #e0e7ff;
  --code-text: #3730a3;
  --success-bg: #f0fdf4;
  --success-border: #10b981;
  --success-text: #047857;
  --danger-bg: #fef2f2;
  --danger-border: #dc2626;
  color-scheme: light;
}

/* Dark mode via class toggle */
[data-theme="dark"] {
  --moodle-accent: #fbbf24;
  --moodle-accent-light: #451a03;
  --moodle-accent-dark: #f59e0b;
  --secondary-accent: #818cf8;
  --secondary-accent-light: #312e81;
  --text-dark: #f1f5f9;
  --text-gray: #94a3b8;
  --text-muted: #64748b;
  --bg-light: #0f172a;
  --bg-white: #1e293b;
  --bg-card: #1e293b;
  --bg-card-hover: #334155;
  --bg-input: #334155;
  --border: #334155;
  --border-light: #475569;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
  --hero-gradient: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
  --info-bg: #451a03;
  --info-border: #f59e0b;
  --info-text: #fde68a;
  --info-strong: #fbbf24;
  --code-bg: #312e81;
  --code-text: #a5b4fc;
  --success-bg: #052e16;
  --success-border: #22c55e;
  --success-text: #4ade80;
  --danger-bg: #450a0a;
  --danger-border: #f87171;
  color-scheme: dark;
}

/* Auto dark mode based on system preference */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --moodle-accent: #fbbf24;
    --moodle-accent-light: #451a03;
    --moodle-accent-dark: #f59e0b;
    --secondary-accent: #818cf8;
    --secondary-accent-light: #312e81;
    --text-dark: #f1f5f9;
    --text-gray: #94a3b8;
    --text-muted: #64748b;
    --bg-light: #0f172a;
    --bg-white: #1e293b;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --bg-input: #334155;
    --border: #334155;
    --border-light: #475569;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --hero-gradient: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    --info-bg: #451a03;
    --info-border: #f59e0b;
    --info-text: #fde68a;
    --info-strong: #fbbf24;
    --code-bg: #312e81;
    --code-text: #a5b4fc;
    --success-bg: #052e16;
    --success-border: #22c55e;
    --success-text: #4ade80;
    --danger-bg: #450a0a;
    --danger-border: #f87171;
    color-scheme: dark;
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg-light);
  color: var(--text-dark);
  line-height: 1.25;
  font-size: 11px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  transition: background-color 0.3s ease, color 0.3s ease;
}

/* Skip to content link for keyboard navigation */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  padding: 8px;
  background: var(--moodle-accent);
  color: white;
  z-index: 100;
  text-decoration: none;
  font-weight: 600;
}

.skip-link:focus {
  top: 0;
}

/* Focus states for accessibility */
a:focus, button:focus, details:focus {
  outline: 2px solid var(--moodle-accent);
  outline-offset: 2px;
}

a:focus-visible {
  outline: 2px solid var(--moodle-accent);
  outline-offset: 2px;
}

.breadcrumbs {
  padding: 4px 10px;
  background: var(--bg-input);
  font-size: 0.75rem;
  color: var(--text-muted);
  transition: background-color 0.3s ease;
}

.breadcrumbs a {
  color: var(--moodle-accent);
  text-decoration: none;
}

.breadcrumbs a:hover {
  text-decoration: underline;
}

.back-to-top {
  display: inline-block;
  margin-bottom: 6px;
  color: var(--moodle-accent);
  text-decoration: none;
  font-size: 0.75rem;
  font-weight: 500;
}

.back-to-top:hover {
  text-decoration: underline;
  color: var(--moodle-accent-dark);
}

.top-nav {
  background: #1e3a5f;
  padding: 6px 12px;
  display: flex;
  justify-content: center;
  gap: 15px;
  flex-wrap: wrap;
}

[data-theme="dark"] .top-nav {
  background: #0f172a;
}

.top-nav a {
  color: white;
  text-decoration: none;
  font-size: 0.8rem;
  font-weight: 500;
  opacity: 0.9;
}

.top-nav a:hover {
  opacity: 1;
  text-decoration: underline;
}

.container {
  display: flex;
  max-width: 1600px;
  margin: 0 auto;
}

.sidebar {
  width: 140px;
  background: var(--bg-white);
  border-right: 1px solid var(--border);
  height: calc(100vh - 24px);
  overflow-y: auto;
  position: sticky;
  top: 0;
  flex-shrink: 0;
  transition: background-color 0.3s ease;
}

.sidebar-header {
  padding: 8px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, var(--moodle-accent-dark) 0%, var(--moodle-accent) 100%);
}

.sidebar-header a {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: white;
}

.sidebar-logo {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  margin-right: 6px;
  background: white;
  padding: 2px;
}

.course-title {
  font-size: 12px;
  font-weight: 700;
}

.sidebar-nav {
  padding: 4px 0;
}

.sidebar-section {
  border-bottom: 1px solid var(--border-light);
}

.sidebar-section summary {
  padding: 6px 10px;
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-dark);
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: background-color 0.2s ease;
}

.sidebar-section summary::-webkit-details-marker {
  display: none;
}

.sidebar-section summary::after {
  content: "+";
  font-size: 12px;
  color: var(--text-gray);
}

.sidebar-section[open] summary::after {
  content: "-";
}

.sidebar-section summary:hover {
  background: var(--bg-input);
}

.sidebar-section ul {
  list-style: none;
  padding: 0 0 6px 0;
  margin: 0;
}

.sidebar-section li {
  margin: 0;
}

.sidebar-section a {
  display: block;
  padding: 3px 8px 3px 16px;
  color: var(--text-gray);
  text-decoration: none;
  font-size: 11px;
  border-left: 2px solid transparent;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.sidebar-section a:hover {
  background: var(--bg-input);
  color: var(--moodle-accent);
}

.sidebar-section a.active {
  border-left-color: var(--moodle-accent);
  color: var(--moodle-accent);
  background: var(--moodle-accent-light);
  font-weight: 600;
}

.main-content {
  flex: 1;
  min-width: 0;
}

.hero {
  background: var(--hero-gradient);
  color: white;
  padding: 6px 8px;
  text-align: center;
  transition: background 0.3s ease;
}

.hero h1 {
  font-size: 1rem;
  margin-bottom: 2px;
  font-weight: 600;
}

.hero .subtitle {
  font-size: 0.7rem;
  opacity: 0.9;
  margin-bottom: 4px;
}

.stats {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 4px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 0.95rem;
  font-weight: 700;
  display: block;
}

.stat-label {
  font-size: 0.6rem;
  opacity: 0.85;
}

.content-area {
  padding: 6px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
}

.content-area .section.full-width {
  grid-column: 1 / -1;
}

.section {
  background: var(--bg-card);
  border-radius: 4px;
  padding: 6px;
  border: 1px solid var(--border);
  transition: background-color 0.3s ease;
}

.section h2 {
  color: var(--text-dark);
  font-size: 0.85rem;
  margin-bottom: 4px;
  border-bottom: 1px solid var(--moodle-accent);
  padding-bottom: 2px;
  font-weight: 600;
}

[data-theme="dark"] .section h2 {
  color: #e2e8f0;
}

.section h3 {
  font-size: 0.75rem;
  color: var(--text-gray);
  font-weight: 600;
  margin-bottom: 4px;
}

.schedule-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 3px;
}

.schedule-item {
  display: flex;
  align-items: center;
  padding: 4px 6px;
  background: var(--bg-input);
  border-radius: 3px;
  border: 1px solid var(--border);
  border-left: 2px solid var(--moodle-accent);
}

.schedule-item.mandatory {
  border-left-color: var(--danger-border);
  background: var(--danger-bg);
}

.schedule-date {
  font-weight: 600;
  min-width: 55px;
  color: var(--text-dark);
  font-size: 0.7rem;
}

.schedule-topic {
  flex: 1;
  font-weight: 500;
  font-size: 0.7rem;
}

.schedule-desc {
  color: var(--text-gray);
  font-size: 0.6rem;
  display: none;
}

.assignment-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
}

.assignment-card {
  background: var(--bg-input);
  padding: 5px;
  border-radius: 3px;
  border: 1px solid var(--border);
}

.assignment-card h3 {
  color: var(--text-dark);
  font-size: 0.7rem;
  margin-bottom: 2px;
}

.assignment-weight {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--moodle-accent);
  display: block;
}

.assignment-type {
  display: inline-block;
  background: var(--moodle-accent-light);
  color: var(--moodle-accent-dark);
  padding: 1px 4px;
  border-radius: 2px;
  font-size: 0.55rem;
  font-weight: 600;
}

.pdf-list {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 3px;
}

.pdf-item {
  display: flex;
  align-items: center;
  padding: 3px 5px;
  background: var(--bg-input);
  border-radius: 3px;
  border: 1px solid var(--border);
  text-decoration: none;
  color: inherit;
}

.pdf-item:hover {
  background: var(--bg-card-hover);
}

.pdf-icon {
  width: 18px;
  height: 18px;
  background: #dc2626;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 2px;
  margin-right: 4px;
  font-weight: 600;
  font-size: 0.5rem;
}

.pdf-info {
  flex: 1;
  min-width: 0;
}

.pdf-title {
  font-weight: 500;
  font-size: 0.65rem;
  display: block;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pdf-section {
  font-size: 0.55rem;
  color: var(--text-gray);
  display: none;
}

.notebook-list {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 3px;
}

.notebook-item {
  display: flex;
  align-items: center;
  padding: 4px 6px;
  background: var(--bg-input);
  border-radius: 3px;
  border: 1px solid var(--border);
  text-decoration: none;
  color: inherit;
}

.notebook-item:hover {
  background: var(--bg-card-hover);
}

.notebook-icon {
  width: 18px;
  height: 18px;
  background: #f59e0b;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 2px;
  margin-right: 5px;
  font-weight: 600;
  font-size: 0.55rem;
}

.notebook-info {
  flex: 1;
}

.notebook-title {
  font-weight: 500;
  font-size: 0.7rem;
  display: block;
}

.notebook-section {
  font-size: 0.6rem;
  color: var(--text-gray);
}

.cta-button {
  display: inline-block;
  padding: 3px 8px;
  background: white;
  color: var(--moodle-accent-dark);
  border-radius: 3px;
  text-decoration: none;
  font-weight: 600;
  font-size: 0.65rem;
  margin: 2px;
}

.cta-button:hover {
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}

.footer {
  text-align: center;
  padding: 4px;
  color: var(--text-gray);
  font-size: 0.6rem;
  border-top: 1px solid var(--border);
  margin-top: 6px;
}

/* Theme Toggle Button */
.theme-toggle {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 6px;
  cursor: pointer;
  padding: 6px 10px;
  color: white;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s ease;
}

.theme-toggle:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(255,255,255,0.5);
}

.theme-toggle-icon {
  font-size: 1rem;
}

/* Sidebar theme toggle (alternative location) */
.sidebar-theme-toggle {
  display: flex;
  justify-content: center;
  padding: 6px 8px;
  border-top: 1px solid var(--border-light);
}

.sidebar-theme-toggle button {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 4px 8px;
  cursor: pointer;
  color: var(--text-gray);
  font-size: 0.7rem;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all 0.2s ease;
  width: 100%;
  justify-content: center;
}

.sidebar-theme-toggle button:hover {
  background: var(--bg-card-hover);
  color: var(--moodle-accent);
}

/* Search Box */
.sidebar-search {
  padding: 6px 8px;
  border-bottom: 1px solid var(--border-light);
}

.sidebar-search input {
  width: 100%;
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-input);
  color: var(--text-dark);
  font-size: 0.75rem;
  transition: all 0.2s ease;
}

.sidebar-search input:focus {
  outline: none;
  border-color: var(--moodle-accent);
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.1);
}

.sidebar-search input::placeholder {
  color: var(--text-muted);
}

/* Search Results */
.search-results {
  padding: 4px 8px;
  max-height: 200px;
  overflow-y: auto;
  display: none;
}

.search-results.active {
  display: block;
}

.search-result-item {
  padding: 4px 6px;
  border-radius: 3px;
  margin-bottom: 2px;
  background: var(--bg-input);
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.search-result-item:hover {
  background: var(--moodle-accent-light);
}

.search-result-item a {
  text-decoration: none;
  color: var(--text-dark);
  display: block;
}

.search-result-title {
  font-weight: 500;
  font-size: 0.75rem;
  margin-bottom: 1px;
}

.search-result-type {
  font-size: 0.6rem;
  color: var(--text-muted);
  text-transform: uppercase;
}

.search-highlight {
  background: var(--moodle-accent-light);
  padding: 0 2px;
  border-radius: 2px;
}

.no-results {
  padding: 6px;
  color: var(--text-muted);
  font-size: 0.75rem;
  text-align: center;
}

@media (max-width: 900px) {
  .container {
    flex-direction: column;
  }
  .sidebar {
    width: 100%;
    height: auto;
    position: relative;
    border-right: none;
    border-bottom: 1px solid var(--border);
    max-height: 40vh;
  }
  .assignment-grid {
    grid-template-columns: 1fr;
  }
  .pdf-list {
    grid-template-columns: 1fr;
  }
}
'''

def get_moodle_head(title):
    """Generate HTML head with Moodle styling"""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | {MOODLE_CONFIG['course_short']}</title>
  <style>
{MOODLE_STYLES}
  </style>
  <script>
    // Theme toggle functionality with localStorage persistence
    (function() {{
      // Check for saved theme preference or system preference
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme) {{
        document.documentElement.setAttribute('data-theme', savedTheme);
      }}
    }})();

    function toggleTheme() {{
      const html = document.documentElement;
      const currentTheme = html.getAttribute('data-theme');
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

      let newTheme;
      if (currentTheme === 'dark') {{
        newTheme = 'light';
      }} else if (currentTheme === 'light') {{
        newTheme = 'dark';
      }} else {{
        // No explicit theme set, toggle based on system preference
        newTheme = systemPrefersDark ? 'light' : 'dark';
      }}

      html.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      updateToggleIcon();
    }}

    function updateToggleIcon() {{
      const html = document.documentElement;
      const currentTheme = html.getAttribute('data-theme');
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const isDark = currentTheme === 'dark' || (currentTheme !== 'light' && systemPrefersDark);

      const toggleBtns = document.querySelectorAll('.theme-toggle-icon');
      toggleBtns.forEach(btn => {{
        btn.textContent = isDark ? 'Light' : 'Dark';
      }});
    }}

    document.addEventListener('DOMContentLoaded', updateToggleIcon);

    // Search functionality
    const searchIndex = [
      // Pages
      {{type: 'Page', title: 'Home', url: 'index.html', keywords: 'overview course nlp text analytics home'}},
      {{type: 'Page', title: 'Schedule', url: 'schedule.html', keywords: 'schedule timeline sessions dates calendar'}},
      {{type: 'Page', title: 'PDFs', url: 'pdfs.html', keywords: 'pdfs slides lectures download materials'}},
      {{type: 'Page', title: 'Notebooks', url: 'notebooks.html', keywords: 'notebooks jupyter python code exercises labs'}},
      {{type: 'Page', title: 'Assignments', url: 'assignments.html', keywords: 'assignments grading assessment presentation'}},
      // Topics
      {{type: 'Topic', title: 'N-Grams', url: 'schedule.html', keywords: 'ngrams language models probability shakespeare'}},
      {{type: 'Topic', title: 'Word Embeddings', url: 'schedule.html', keywords: 'embeddings word2vec vectors semantic'}},
      {{type: 'Topic', title: 'Neural Networks', url: 'schedule.html', keywords: 'neural networks backpropagation perceptron'}},
      {{type: 'Topic', title: 'RNNs', url: 'schedule.html', keywords: 'rnn recurrent neural networks sequence'}},
      {{type: 'Topic', title: 'LSTMs', url: 'schedule.html', keywords: 'lstm long short term memory gates'}},
      {{type: 'Topic', title: 'Transformers', url: 'schedule.html', keywords: 'transformer attention self-attention bert gpt'}},
      {{type: 'Topic', title: 'Sequence-to-Sequence', url: 'schedule.html', keywords: 'seq2seq encoder decoder translation'}},
      {{type: 'Topic', title: 'Decoding Strategies', url: 'schedule.html', keywords: 'decoding greedy beam search sampling nucleus'}},
      {{type: 'Topic', title: 'Fine-tuning', url: 'schedule.html', keywords: 'finetuning transfer learning bert'}},
      {{type: 'Topic', title: 'Ethics', url: 'schedule.html', keywords: 'ethics bias fairness responsible ai'}}
    ];

    function performSearch(query) {{
      if (!query || query.length < 2) return [];
      const lowerQuery = query.toLowerCase();
      return searchIndex.filter(item =>
        item.title.toLowerCase().includes(lowerQuery) ||
        item.keywords.toLowerCase().includes(lowerQuery)
      ).slice(0, 8);
    }}

    function displayResults(results, query) {{
      const resultsContainer = document.getElementById('search-results');
      if (!resultsContainer) return;

      if (results.length === 0) {{
        resultsContainer.innerHTML = '<div class="no-results">No results found</div>';
        resultsContainer.classList.add('active');
        return;
      }}

      resultsContainer.innerHTML = results.map(item => `
        <div class="search-result-item">
          <a href="${{item.url}}">
            <div class="search-result-title">${{item.title}}</div>
            <div class="search-result-type">${{item.type}}</div>
          </a>
        </div>
      `).join('');
      resultsContainer.classList.add('active');
    }}

    function handleSearch(e) {{
      const query = e.target.value;
      const resultsContainer = document.getElementById('search-results');

      if (query.length < 2) {{
        if (resultsContainer) resultsContainer.classList.remove('active');
        return;
      }}

      const results = performSearch(query);
      displayResults(results, query);
    }}

    document.addEventListener('DOMContentLoaded', function() {{
      const searchInput = document.getElementById('site-search');
      if (searchInput) {{
        searchInput.addEventListener('input', handleSearch);
        searchInput.addEventListener('focus', function() {{
          if (this.value.length >= 2) {{
            const results = performSearch(this.value);
            displayResults(results, this.value);
          }}
        }});
      }}

      // Close search results when clicking outside
      document.addEventListener('click', function(e) {{
        const searchContainer = document.querySelector('.sidebar-search');
        const resultsContainer = document.getElementById('search-results');
        if (searchContainer && resultsContainer && !searchContainer.contains(e.target)) {{
          resultsContainer.classList.remove('active');
        }}
      }});
    }});
  </script>
</head>'''

def get_breadcrumbs(current_page, page_title):
    """Generate breadcrumb navigation"""
    return f'''  <nav class="breadcrumbs" id="top">
    <a href="index.html">Home</a> &gt; <span>{page_title}</span>
  </nav>
'''

def get_moodle_nav(active_page=''):
    """Generate top navigation"""
    nav_items = [
        ('index.html', 'Home'),
        ('schedule.html', 'Schedule'),
        ('pdfs.html', 'PDFs'),
        ('notebooks.html', 'Notebooks'),
        ('assignments.html', 'Assignments')
    ]

    nav_html = '  <nav class="top-nav" aria-label="Main navigation">\n'
    for href, label in nav_items:
        active_class = ' class="active"' if href == active_page else ''
        nav_html += f'    <a href="{href}"{active_class}>{label}</a>\n'
    nav_html += f'    <a href="{MOODLE_CONFIG["streamlit_app"]}" target="_blank" rel="noopener noreferrer">NLP Evolution App</a>\n'
    nav_html += f'    <a href="{MOODLE_CONFIG["github_url"]}" target="_blank" rel="noopener noreferrer">GitHub Repository</a>\n'
    nav_html += '  </nav>\n'
    return nav_html

def get_moodle_sidebar(active_page=''):
    """Generate sidebar with Moodle sections"""
    nav_items = [
        ('index.html', 'Overview'),
        ('schedule.html', 'Schedule'),
        ('pdfs.html', 'PDFs'),
        ('notebooks.html', 'Notebooks'),
        ('assignments.html', 'Assignments')
    ]

    sidebar_html = '''    <aside class="sidebar">
      <div class="sidebar-header">
        <a href="index.html">
          <img src="https://quantlet.com/images/Q.png" alt="NLP Course Logo - QuantLet" class="sidebar-logo">
          <span class="course-title">''' + MOODLE_CONFIG['course_short'] + '''</span>
        </a>
      </div>
      <div class="sidebar-search">
        <input type="text" id="site-search" placeholder="Search course..." aria-label="Search course content">
        <div id="search-results" class="search-results"></div>
      </div>
      <nav class="sidebar-nav" aria-label="Course navigation">
        <details class="sidebar-section" open>
          <summary>Navigation</summary>
          <ul>
'''

    for href, label in nav_items:
        active_class = ' class="active"' if href == active_page else ''
        sidebar_html += f'            <li><a href="{href}"{active_class}>{label}</a></li>\n'

    sidebar_html += '''          </ul>
        </details>
        <details class="sidebar-section">
          <summary>External Links</summary>
          <ul>
            <li><a href="''' + MOODLE_CONFIG['streamlit_app'] + '''" target="_blank" rel="noopener noreferrer">NLP Evolution App</a></li>
            <li><a href="''' + MOODLE_CONFIG['github_url'] + '''" target="_blank" rel="noopener noreferrer">GitHub Repository</a></li>
          </ul>
        </details>
        <div class="sidebar-theme-toggle">
          <button onclick="toggleTheme()" aria-label="Toggle dark mode">
            <span class="theme-toggle-icon">Dark</span> Mode
          </button>
        </div>
      </nav>
    </aside>
'''
    return sidebar_html

def get_moodle_footer():
    """Generate footer"""
    return '''      <footer class="footer">
        <a href="#top" class="back-to-top">&#8593; Back to Top</a>
        <p>&copy; 2025 NLP Course | ''' + MOODLE_CONFIG['course_name'] + '''</p>
      </footer>'''

def load_data():
    """Load Moodle data and asset manifest"""
    scripts_dir = Path(__file__).parent

    # Load moodle_data.json
    moodle_data_path = scripts_dir / 'moodle_data.json'
    with open(moodle_data_path, 'r', encoding='utf-8') as f:
        moodle_data = json.load(f)

    # Load manifest.json
    manifest_path = scripts_dir.parent.parent / 'docs' / 'moodle' / 'assets' / 'manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    return moodle_data, manifest

def generate_index_page(moodle_data, manifest):
    """Generate overview page with hero section and quick links"""
    stats = manifest['statistics']

    html = get_moodle_head('Home')
    html += '<body>\n'
    html += '  <a href="#main-content" class="skip-link">Skip to main content</a>\n'
    html += get_moodle_nav('index.html')
    html += '  <div class="container">\n'
    html += get_moodle_sidebar('index.html')
    html += '''    <main class="main-content" id="main-content">
      <div class="hero">
        <h1>''' + MOODLE_CONFIG['course_name'] + '''</h1>
        <p class="subtitle">Natural Language Processing and Text Analytics</p>
        <div class="stats">
          <div class="stat-item">
            <span class="stat-number">''' + str(stats['total_pdfs']) + '''</span>
            <span class="stat-label">PDFs</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">''' + str(stats['total_notebooks']) + '''</span>
            <span class="stat-label">Notebooks</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">13</span>
            <span class="stat-label">Sessions</span>
          </div>
        </div>
        <div style="margin-top: 20px;">
          <a href="''' + MOODLE_CONFIG['github_url'] + '''" class="cta-button" target="_blank">GitHub Repository</a>
          <a href="''' + MOODLE_CONFIG['streamlit_app'] + '''" class="cta-button" target="_blank">NLP Evolution App</a>
        </div>
      </div>
      <div class="content-area">
        <div class="section">
          <h2>Course Overview</h2>
          <p style="margin-bottom: 15px; line-height: 1.6; color: var(--text-gray);">
            This hands-on course takes you from statistical language models to modern transformer architectures
            through discovery-based learning. You will build practical NLP systems using PyTorch while understanding
            the mathematical foundations that power today's large language models.
          </p>
          <h3>What You'll Learn</h3>
          <ul style="margin-left: 20px; line-height: 1.8; color: var(--text-gray);">
            <li>Build neural language models from scratch using PyTorch</li>
            <li>Master transformer architectures and attention mechanisms</li>
            <li>Fine-tune pre-trained models for sentiment analysis and text classification</li>
            <li>Implement decoding strategies for controllable text generation</li>
            <li>Understand efficiency techniques: quantization, pruning, and knowledge distillation</li>
            <li>Apply ethical AI principles to NLP systems</li>
          </ul>
          <div style="margin-top: 15px; padding: 12px; background: var(--info-bg); border-left: 4px solid var(--info-border); border-radius: 4px;">
            <strong style="color: var(--info-strong);">Prerequisites:</strong>
            <span style="color: var(--info-text);">Programming experience in Python. No prior deep learning knowledge required. Mathematics: basic calculus and linear algebra helpful but not mandatory.</span>
          </div>
        </div>

        <!-- Learning Path Section -->
        <div class="section">
          <h2>Learning Path</h2>
          <p style="margin-bottom: 15px; line-height: 1.6; color: var(--text-gray);">
            The course follows a progressive structure with three main phases:
          </p>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
            <div style="padding: 15px; background: var(--success-bg); border-left: 4px solid var(--success-border); border-radius: 6px;">
              <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: var(--success-border); color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Phase 1</span>
                <strong style="color: var(--success-text);">Foundations</strong>
              </div>
              <p style="font-size: 0.9rem; color: var(--text-gray); margin-bottom: 8px;">Weeks 1-5</p>
              <ul style="margin-left: 15px; font-size: 0.85rem; color: var(--text-gray);">
                <li>N-Grams & Statistical Models</li>
                <li>Word Embeddings</li>
                <li>Neural Networks Primer</li>
                <li>RNNs & LSTMs</li>
              </ul>
            </div>
            <div style="padding: 15px; background: var(--info-bg); border-left: 4px solid var(--info-border); border-radius: 6px;">
              <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: var(--moodle-accent); color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Phase 2</span>
                <strong style="color: var(--info-strong);">Core Architectures</strong>
              </div>
              <p style="font-size: 0.9rem; color: var(--text-gray); margin-bottom: 8px;">Weeks 6-9</p>
              <ul style="margin-left: 15px; font-size: 0.85rem; color: var(--text-gray);">
                <li>Sequence-to-Sequence</li>
                <li>Transformers</li>
                <li>Multi-Agent LLMs</li>
                <li>Decoding Strategies</li>
              </ul>
            </div>
            <div style="padding: 15px; background: var(--secondary-accent-light); border-left: 4px solid var(--secondary-accent); border-radius: 6px;">
              <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: var(--secondary-accent); color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Phase 3</span>
                <strong style="color: var(--secondary-accent);">Applications</strong>
              </div>
              <p style="font-size: 0.9rem; color: var(--text-gray); margin-bottom: 8px;">Weeks 10-13</p>
              <ul style="margin-left: 15px; font-size: 0.85rem; color: var(--text-gray);">
                <li>Fine-tuning & Transfer</li>
                <li>Efficiency & Deployment</li>
                <li>Ethics in NLP</li>
                <li>Final Projects</li>
              </ul>
            </div>
          </div>
        </div>

        <div class="section">
          <h2>Quick Links</h2>
          <div class="assignment-grid">
            <a href="schedule.html" style="text-decoration: none; color: inherit;">
              <div class="assignment-card">
                <h3>Schedule</h3>
                <p style="color: var(--text-gray);">View 13-week course timeline with mandatory sessions</p>
              </div>
            </a>
            <a href="pdfs.html" style="text-decoration: none; color: inherit;">
              <div class="assignment-card">
                <h3>PDFs</h3>
                <p style="color: var(--text-gray);">Download ''' + str(stats['total_pdfs']) + ''' lecture slides and exercises</p>
              </div>
            </a>
            <a href="notebooks.html" style="text-decoration: none; color: inherit;">
              <div class="assignment-card">
                <h3>Notebooks</h3>
                <p style="color: var(--text-gray);">Access ''' + str(stats['total_notebooks']) + ''' interactive Jupyter notebooks</p>
              </div>
            </a>
            <a href="assignments.html" style="text-decoration: none; color: inherit;">
              <div class="assignment-card">
                <h3>Assignments</h3>
                <p style="color: var(--text-gray);">Assessment overview and grading weights</p>
              </div>
            </a>
          </div>
        </div>

        <!-- FAQ Section -->
        <div class="section">
          <h2>Frequently Asked Questions</h2>
          <div style="display: grid; gap: 8px;">
            <details style="background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 0;">
              <summary style="padding: 12px 15px; cursor: pointer; font-weight: 500; color: var(--text-dark); list-style: none; display: flex; justify-content: space-between; align-items: center;">
                What programming experience do I need?
                <span style="color: var(--moodle-accent);">+</span>
              </summary>
              <div style="padding: 0 15px 15px 15px; color: var(--text-gray); font-size: 0.95rem;">
                Intermediate Python experience is required. You should be comfortable with functions, classes, lists, dictionaries, and file I/O. Experience with NumPy is helpful but not required - we'll cover the basics needed for deep learning.
              </div>
            </details>
            <details style="background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 0;">
              <summary style="padding: 12px 15px; cursor: pointer; font-weight: 500; color: var(--text-dark); list-style: none; display: flex; justify-content: space-between; align-items: center;">
                What software do I need to install?
                <span style="color: var(--moodle-accent);">+</span>
              </summary>
              <div style="padding: 0 15px 15px 15px; color: var(--text-gray); font-size: 0.95rem;">
                Python 3.8+, PyTorch, NumPy, Matplotlib, Jupyter Lab. Alternatively, you can use Google Colab for zero-setup cloud execution. Detailed installation instructions are provided in the first session.
              </div>
            </details>
            <details style="background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 0;">
              <summary style="padding: 12px 15px; cursor: pointer; font-weight: 500; color: var(--text-dark); list-style: none; display: flex; justify-content: space-between; align-items: center;">
                How is the course graded?
                <span style="color: var(--moodle-accent);">+</span>
              </summary>
              <div style="padding: 0 15px 15px 15px; color: var(--text-gray); font-size: 0.95rem;">
                Three assessments: Kurzprasentation (20%, individual 5-min talk), Zwischenprasentation (10%, team mid-term), and Abschlussprasentation (70%, team final project). No written exams. See the Assignments page for details.
              </div>
            </details>
            <details style="background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 0;">
              <summary style="padding: 12px 15px; cursor: pointer; font-weight: 500; color: var(--text-dark); list-style: none; display: flex; justify-content: space-between; align-items: center;">
                Can I access materials after the course ends?
                <span style="color: var(--moodle-accent);">+</span>
              </summary>
              <div style="padding: 0 15px 15px 15px; color: var(--text-gray); font-size: 0.95rem;">
                Yes! All course materials are available on GitHub and will remain accessible indefinitely. Lecture slides, notebooks, and exercises can be downloaded for offline use.
              </div>
            </details>
          </div>
        </div>
      </div>
'''
    html += get_moodle_footer()
    html += '''    </main>
  </div>
</body>
</html>'''
    return html

def generate_schedule_page():
    """Generate schedule page with timeline view"""
    # Learning objectives for each session
    learning_objectives = {
        'Shakespeare und N-Grams': 'Understand probabilistic language models and build a text generator using n-gram statistics',
        'Word Embeddings': 'Learn how words can be represented as vectors and explore semantic relationships in embedding space',
        'Neuronale Netze': 'Master neural network fundamentals: forward propagation, backpropagation, and universal function approximation',
        'RNNs': 'Discover how recurrent architectures process sequential data and handle variable-length inputs',
        'LSTMs': 'Understand the vanishing gradient problem and how LSTM gates enable long-term memory',
        'Zwischenprasentation': 'Present your understanding of 4 NLP methods to demonstrate mid-course progress',
        'Sequence-to-Sequence': 'Build encoder-decoder models for translation and learn attention mechanisms',
        'Transformers': 'Master self-attention, multi-head attention, and the transformer architecture',
        'Multi-Agent LLMs': 'Explore how multiple language models can collaborate to solve complex tasks',
        'Kurzprasentation Option 1': 'Share deep dive into one course topic with the class',
        'Kurzprasentation Option 2': 'Share deep dive into one course topic with the class',
        'Decoding Strategies': 'Compare greedy, beam search, sampling, and nucleus sampling for text generation',
        'Abschlussprasentation': 'Present final project demonstrating NLP techniques learned throughout the course'
    }

    html = get_moodle_head('Schedule')
    html += '<body>\n'
    html += '  <a href="#main-content" class="skip-link">Skip to main content</a>\n'
    html += get_moodle_nav('schedule.html')
    html += get_breadcrumbs('schedule.html', 'Schedule')
    html += '  <div class="container">\n'
    html += get_moodle_sidebar('schedule.html')
    html += '''    <main class="main-content" id="main-content">
      <div class="hero">
        <h1>Course Schedule</h1>
        <p class="subtitle">13 Sessions | September - December 2025</p>
      </div>
      <div class="content-area">
        <!-- Progress Overview -->
        <div class="section">
          <h2>Course Progress</h2>
          <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <div style="flex: 1; background: var(--border); border-radius: 10px; height: 12px; overflow: hidden;">
              <div style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--success-border), var(--moodle-accent)); border-radius: 10px; transition: width 0.5s ease;" id="progress-bar"></div>
            </div>
            <span style="font-weight: 600; color: var(--text-dark); min-width: 50px;" id="progress-text">0%</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-gray);">
            <span>0 of 13 sessions completed</span>
            <span>Course starts Sep 12, 2025</span>
          </div>
        </div>

        <div class="section">
          <p style="margin-bottom: 15px; line-height: 1.6; color: var(--text-gray);">
            The course follows a progressive structure from foundational statistical models to cutting-edge neural architectures.
            Each session builds on previous concepts through hands-on coding exercises and theoretical deep dives.
            <strong style="color: var(--danger-border);">Red-bordered sessions are mandatory presentations.</strong>
          </p>
          <div style="margin-bottom: 15px; padding: 12px; background: var(--info-bg); border-left: 4px solid var(--info-border); border-radius: 4px;">
            <strong style="color: var(--info-strong);">Course Structure:</strong>
            <span style="color: var(--info-text);">Weeks 1-5 cover foundations (N-grams to LSTMs), Weeks 6-9 explore transformers and advanced architectures, Weeks 10-13 focus on practical applications and optimization.</span>
          </div>
        </div>
        <div class="section">
          <h2>Timeline</h2>
          <div class="schedule-grid">
'''

    for session in SCHEDULE:
        mandatory_class = ' mandatory' if session['mandatory'] else ''
        objective = learning_objectives.get(session['topic'], '')
        html += f'''            <div class="schedule-item{mandatory_class}">
              <span class="schedule-date">{session['date']}</span>
              <div>
                <div class="schedule-topic">{session['topic']}</div>
                <div class="schedule-desc">{session['description']}</div>'''

        if objective and not session['mandatory']:
            html += f'''
                <div style="margin-top: 5px; font-size: 0.85rem; color: #475569; font-style: italic;">{objective}</div>'''

        html += '''
              </div>
            </div>
'''

    html += '''          </div>
        </div>
      </div>
'''
    html += get_moodle_footer()
    html += '''    </main>
  </div>
</body>
</html>'''
    return html

def generate_pdfs_page(manifest):
    """Generate PDF download page with grid layout"""
    # Group PDFs by topic area
    topic_groups = {
        'Foundations': ['N-Grams', 'Word Embeddings', 'Neural Networks'],
        'Sequential Models': ['RNN', 'LSTM', 'Sequence-to-Sequence'],
        'Transformers': ['Transformer', 'Pre-trained', 'Advanced'],
        'Applications': ['Tokenization', 'Decoding', 'Fine-tuning', 'Efficiency', 'Ethics'],
        'Special Modules': ['Summarization', 'Sentiment', 'ML Intro']
    }

    html = get_moodle_head('PDFs')
    html += '<body>\n'
    html += '  <a href="#main-content" class="skip-link">Skip to main content</a>\n'
    html += get_moodle_nav('pdfs.html')
    html += get_breadcrumbs('pdfs.html', 'PDFs')
    html += '  <div class="container">\n'
    html += get_moodle_sidebar('pdfs.html')
    html += '''    <main class="main-content" id="main-content">
      <div class="hero">
        <h1>PDF Downloads</h1>
        <p class="subtitle">''' + str(len(manifest['pdfs'])) + ''' Course Materials Available</p>
      </div>
      <div class="content-area">
        <div class="section">
          <p style="margin-bottom: 15px; line-height: 1.6; color: #334155;">
            All lecture slides use discovery-based pedagogy with concrete examples before abstract theory.
            Each PDF includes worked examples, checkpoint quizzes, and BSc Discovery color scheme for clarity.
            Materials progress from statistical foundations through neural architectures to modern transformers.
          </p>
          <div style="margin-bottom: 15px; padding: 12px; background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 4px;">
            <strong style="color: #92400e;">Download Tips:</strong>
            <span style="color: #78350f;">Click any PDF to download. Materials are organized by topic area below. Start with Foundations if new to NLP, or jump to specific topics as needed.</span>
          </div>
        </div>'''

    # Group and display PDFs by topic
    for group_name, keywords in topic_groups.items():
        group_pdfs = [pdf for pdf in manifest['pdfs'] if any(kw.lower() in pdf['section'].lower() or kw.lower() in pdf['title'].lower() for kw in keywords)]

        if group_pdfs:
            html += f'''
        <div class="section">
          <h2>{group_name}</h2>
          <div class="pdf-list">
'''
            for pdf in group_pdfs:
                html += f'''            <a href="assets/{pdf['output_path']}" class="pdf-item" download>
              <div class="pdf-icon">PDF</div>
              <div class="pdf-info">
                <span class="pdf-title">{pdf['title']}</span>
                <span class="pdf-section">{pdf['section']}</span>
              </div>
            </a>
'''
            html += '''          </div>
        </div>'''

    html += '''
      </div>
'''
    html += get_moodle_footer()
    html += '''    </main>
  </div>
</body>
</html>'''
    return html

def generate_notebooks_page(manifest):
    """Generate notebooks gallery page"""
    # Notebook descriptions, technology tags, and difficulty levels
    notebook_info = {
        'week01': {'desc': 'Build probabilistic text generators using n-gram models and explore perplexity metrics', 'tags': ['Python', 'NumPy', 'Probability'], 'difficulty': 'Beginner'},
        'week02': {'desc': 'Train Word2Vec embeddings and visualize semantic relationships in vector space', 'tags': ['Python', 'Gensim', 'Visualization'], 'difficulty': 'Beginner'},
        'week03': {'desc': 'Implement feedforward neural networks from scratch and understand backpropagation', 'tags': ['Python', 'NumPy', 'Neural Networks'], 'difficulty': 'Intermediate'},
        'week04': {'desc': 'Build recurrent neural networks for sequence modeling and text prediction', 'tags': ['Python', 'PyTorch', 'RNN'], 'difficulty': 'Intermediate'},
        'week05': {'desc': 'Implement LSTM networks and compare with vanilla RNNs on long sequences', 'tags': ['Python', 'PyTorch', 'LSTM'], 'difficulty': 'Intermediate'},
        'week07': {'desc': 'Build sequence-to-sequence models with attention for translation tasks', 'tags': ['Python', 'PyTorch', 'Attention'], 'difficulty': 'Advanced'},
        'week08': {'desc': 'Implement transformer architecture from scratch with multi-head attention', 'tags': ['Python', 'PyTorch', 'Transformers'], 'difficulty': 'Advanced'},
        'week09': {'desc': 'Explore text decoding strategies: greedy, beam search, sampling, nucleus sampling', 'tags': ['Python', 'PyTorch', 'Generation'], 'difficulty': 'Intermediate'},
        'week10': {'desc': 'Fine-tune BERT for sentiment analysis and text classification', 'tags': ['Python', 'PyTorch', 'BERT'], 'difficulty': 'Advanced'},
        'embeddings': {'desc': 'Interactive 3D visualization of word embeddings with t-SNE and PCA', 'tags': ['Python', 'Plotly', '3D Viz'], 'difficulty': 'Beginner'},
        'summarization': {'desc': 'Build abstractive and extractive summarization systems', 'tags': ['Python', 'PyTorch', 'Summarization'], 'difficulty': 'Advanced'}
    }

    # Difficulty badge colors
    difficulty_colors = {
        'Beginner': ('var(--success-border)', 'white'),
        'Intermediate': ('var(--moodle-accent)', 'white'),
        'Advanced': ('var(--secondary-accent)', 'white')
    }

    html = get_moodle_head('Notebooks')
    html += '<body>\n'
    html += '  <a href="#main-content" class="skip-link">Skip to main content</a>\n'
    html += get_moodle_nav('notebooks.html')
    html += get_breadcrumbs('notebooks.html', 'Notebooks')
    html += '  <div class="container">\n'
    html += get_moodle_sidebar('notebooks.html')
    html += '''    <main class="main-content" id="main-content">
      <div class="hero">
        <h1>Jupyter Notebooks</h1>
        <p class="subtitle">''' + str(len(manifest['notebooks'])) + ''' Interactive Notebooks</p>
      </div>
      <div class="content-area">
        <div class="section">
          <p style="margin-bottom: 15px; line-height: 1.6; color: #334155;">
            Hands-on Python notebooks for every major topic. Each notebook contains executable code,
            explanatory markdown, and exercises to reinforce learning. Download and run locally in Jupyter Lab
            or upload to Google Colab for cloud execution.
          </p>
          <div style="margin-bottom: 15px; padding: 12px; background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 4px;">
            <strong style="color: #92400e;">Getting Started:</strong>
            <span style="color: #78350f;">Install requirements: pip install torch numpy matplotlib seaborn jupyter. Or use Google Colab for zero-setup cloud execution.</span>
          </div>
        </div>
        <div class="section">
          <h2>All Notebooks</h2>
          <div class="notebook-list">
'''

    for notebook in manifest['notebooks']:
        # Try to match notebook info
        info = None
        for key, data in notebook_info.items():
            if key.lower() in notebook['title'].lower() or key.lower() in notebook['section'].lower():
                info = data
                break

        html += f'''            <a href="assets/{notebook['output_path']}" class="notebook-item" download>
              <div class="notebook-icon">NB</div>
              <div class="notebook-info">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                  <span class="notebook-title">{notebook['title']}</span>'''

        if info and 'difficulty' in info:
            bg_color, text_color = difficulty_colors.get(info['difficulty'], ('var(--text-gray)', 'white'))
            html += f'''
                  <span style="background: {bg_color}; color: {text_color}; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600;">{info['difficulty']}</span>'''

        html += f'''
                </div>
                <span class="notebook-section">{notebook['section']}</span>'''

        if info:
            html += f'''
                <div style="margin-top: 6px; font-size: 0.85rem; color: var(--text-gray);">{info['desc']}</div>
                <div style="margin-top: 4px; display: flex; gap: 5px; flex-wrap: wrap;">'''
            for tag in info['tags']:
                html += f'''
                  <span style="background: var(--code-bg); color: var(--code-text); padding: 2px 6px; border-radius: 3px; font-size: 0.75rem; font-weight: 500;">{tag}</span>'''
            html += '''
                </div>'''

        html += '''
              </div>
            </a>
'''

    html += '''          </div>
        </div>
      </div>
'''
    html += get_moodle_footer()
    html += '''    </main>
  </div>
</body>
</html>'''
    return html

def generate_assignments_page():
    """Generate assessments page with cards"""
    html = get_moodle_head('Assignments')
    html += '<body>\n'
    html += '  <a href="#main-content" class="skip-link">Skip to main content</a>\n'
    html += get_moodle_nav('assignments.html')
    html += get_breadcrumbs('assignments.html', 'Assignments')
    html += '  <div class="container">\n'
    html += get_moodle_sidebar('assignments.html')
    html += '''    <main class="main-content" id="main-content">
      <div class="hero">
        <h1>Assignments</h1>
        <p class="subtitle">3 Assessments | 100% Total Weight</p>
      </div>
      <div class="content-area">
        <div class="section">
          <h2>Assessment Overview</h2>
          <div class="assignment-grid">
'''

    for assignment in ASSIGNMENTS:
        html += f'''            <div class="assignment-card">
              <h3>{assignment['name']}</h3>
              <span class="assignment-weight">{assignment['weight']}%</span>
              <span class="assignment-type">{assignment['type'].upper()}</span>
              <p>{assignment['description']}</p>
            </div>
'''

    html += '''          </div>
        </div>
        <div class="section">
          <h2>Grading Criteria</h2>
          <p style="margin-bottom: 12px; line-height: 1.6; color: #334155;">
            All presentations are assessed on technical understanding, clarity of explanation, and ability to answer questions.
          </p>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
            <div style="padding: 10px; background: #f8fafc; border-radius: 6px; border-left: 3px solid #3b82f6;">
              <strong style="color: #1e40af; display: block; margin-bottom: 5px;">Technical Depth (40%)</strong>
              <span style="font-size: 0.9rem; color: #475569;">Demonstrates mastery of underlying concepts, algorithms, and mathematical foundations</span>
            </div>
            <div style="padding: 10px; background: #f8fafc; border-radius: 6px; border-left: 3px solid #10b981;">
              <strong style="color: #047857; display: block; margin-bottom: 5px;">Communication (30%)</strong>
              <span style="font-size: 0.9rem; color: #475569;">Clear explanations, effective visuals, logical flow, engagement with audience</span>
            </div>
            <div style="padding: 10px; background: #f8fafc; border-radius: 6px; border-left: 3px solid #f59e0b;">
              <strong style="color: #d97706; display: block; margin-bottom: 5px;">Q&A Performance (30%)</strong>
              <span style="font-size: 0.9rem; color: #475569;">Thoughtful responses, admits unknowns appropriately, demonstrates flexibility</span>
            </div>
          </div>
        </div>
        <div class="section">
          <h2>Tips for Success</h2>
          <ul style="margin-left: 20px; line-height: 1.8; color: #334155;">
            <li><strong>Kurzprasentation:</strong> Focus on depth over breadth. Pick one concept and explain it thoroughly with concrete examples.</li>
            <li><strong>Zwischenprasentation:</strong> Show comparative understanding. Explain how 4 methods differ and when to use each.</li>
            <li><strong>Abschlussprasentation:</strong> Demonstrate integration of course concepts. Include live demo or detailed walkthrough of your implementation.</li>
            <li><strong>All presentations:</strong> Practice timing, prepare for questions, use visual aids effectively, cite sources appropriately.</li>
          </ul>
        </div>
        <div class="section">
          <h2>Submission Guidelines</h2>
          <div style="padding: 12px; background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 4px; margin-bottom: 12px;">
            <strong style="color: #92400e;">Required Materials:</strong>
            <span style="color: #78350f;">Upload presentation slides (PDF) to Moodle at least 24 hours before your session. Include references and code snippets where applicable.</span>
          </div>
          <p style="line-height: 1.6; color: #334155;">
            <strong>Format:</strong> Use 16:9 aspect ratio. File naming: LastName_FirstName_AssignmentType.pdf<br>
            <strong>Code:</strong> If presenting technical implementation, include GitHub repository link or code appendix.<br>
            <strong>Timing:</strong> Stick to allocated time limits. Practice with timer to ensure coverage of all key points.
          </p>
        </div>
      </div>
'''
    html += get_moodle_footer()
    html += '''    </main>
  </div>
</body>
</html>'''
    return html

def main():
    """Main generation orchestrator"""
    print("Moodle Site Generator")
    print("=" * 50)

    # Load data
    print("\nLoading data...")
    moodle_data, manifest = load_data()

    # Print loaded data summary
    print(f"\nCourse name: {moodle_data['course_name']}")
    print(f"Number of sections: {len(moodle_data['sections'])}")
    print(f"Number of PDFs in manifest: {len(manifest['pdfs'])}")
    print(f"Number of notebooks: {len(manifest['notebooks'])}")

    # Create output directory
    output_dir = MOODLE_CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate all pages
    print("\nGenerating pages...")
    pages = {
        'index.html': generate_index_page(moodle_data, manifest),
        'schedule.html': generate_schedule_page(),
        'pdfs.html': generate_pdfs_page(manifest),
        'notebooks.html': generate_notebooks_page(manifest),
        'assignments.html': generate_assignments_page()
    }

    # Write all HTML files
    print("\nWriting HTML files...")
    for filename, html_content in pages.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        file_size = output_path.stat().st_size
        print(f"  {filename}: {file_size:,} bytes")

    print(f"\nAll pages generated successfully!")
    print(f"Total files: {len(pages)}")
    print(f"\nView site at: {output_dir / 'index.html'}")

if __name__ == '__main__':
    main()
