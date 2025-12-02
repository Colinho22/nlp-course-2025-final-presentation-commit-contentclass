/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // BSc Discovery / Madrid Beamer Color Palette (EXACT from template_beamer_final.tex)
        mlblue: '#0066CC',        // RGB(0,102,204)
        mlpurple: '#3333B2',      // RGB(51,51,178) - PRIMARY BRAND
        mllavender: '#ADADE0',    // RGB(173,173,224)
        mllavender2: '#C1C1E8',   // RGB(193,193,232)
        mllavender3: '#CCCCEB',   // RGB(204,204,235) - Main backgrounds
        mllavender4: '#D6D6EF',   // RGB(214,214,239) - Block backgrounds
        mlorange: '#FF7F0E',      // RGB(255,127,14)
        mlgreen: '#2CA02C',       // RGB(44,160,44)
        mlred: '#D62728',         // RGB(214,39,40)
        mlgray: '#7F7F7F',        // RGB(127,127,127)
        lightgray: '#F0F0F0',     // RGB(240,240,240)
        midgray: '#B4B4B4',       // RGB(180,180,180)
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Monaco', 'Courier New', 'monospace'],
      },
      fontSize: {
        '8pt': ['8pt', { lineHeight: '1.2' }],
        '11pt': ['11pt', { lineHeight: '1.4' }],
      },
      aspectRatio: {
        'beamer': '16 / 9',
      },
    },
  },
  plugins: [],
}
