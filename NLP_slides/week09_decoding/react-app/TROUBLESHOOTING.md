# Troubleshooting Guide

## Server Status
- ✅ Server running at: http://localhost:5178/
- ✅ Compilation successful (no errors in terminal)
- ✅ All files present

## If the page is blank or shows errors:

### Step 1: Check Browser Console
1. Open http://localhost:5178/
2. Press F12 to open Developer Tools
3. Go to "Console" tab
4. Look for red error messages
5. Copy the error message

### Step 2: Common Issues

**Issue: "Cannot find module"**
- Solution: Check that all imports use correct paths
- Verify: `week09_slides_complete.json` exists in `src/data/`

**Issue: "Failed to load PDF"**
- Solution: Check `public/figures/` contains PDF files
- Verify: 67 PDF files copied

**Issue: "KaTeX error"**
- Solution: Check math formula syntax
- Some formulas may need adjustment

**Issue: Blank page, no errors**
- Solution: Check App.jsx is rendering Presentation component
- Verify: `import Presentation from './components/Presentation/Presentation'`

### Step 3: Verify File Structure

Run this:
```bash
cd NLP_slides/week09_decoding/react-app
ls -la src/components/Presentation/
ls -la src/data/
ls -la public/figures/ | head -10
```

Should show:
- Presentation.jsx, Controls.jsx, etc. in Presentation/
- week09_slides_complete.json in data/
- Multiple .pdf files in figures/

### Step 4: Check Network Tab
1. F12 → Network tab
2. Reload page (Ctrl+R)
3. Look for failed requests (red)
4. Common issues:
   - 404 for figures → PDFs not copied
   - 404 for JSON → Data file missing
   - CORS errors → PDF.js worker issue

### Step 5: Try Simple Test

Replace App.jsx content temporarily with:
```jsx
function App() {
  return <div className="p-10 text-center">
    <h1 className="text-4xl font-bold text-mlpurple">
      Test - If you see this, React is working!
    </h1>
  </div>;
}

export default App;
```

If this works, the issue is in Presentation component.
If this doesn't work, there's a build configuration issue.

## Specific Error Messages

### "slideData.metadata is undefined"
**Cause**: JSON structure mismatch
**Fix**: Check JSON has `metadata` and `slides` keys

### "Cannot read property 'sections' of undefined"
**Cause**: JSON metadata missing sections
**Fix**: Re-run `python extract_slides_enhanced.py`

### "Failed to parse source for import analysis"
**Cause**: JSX in .js file
**Fix**: Rename to .jsx extension

### PDF.js errors
**Cause**: Worker not loading
**Fix**: Check internet connection (worker loads from CDN)

## Still Not Working?

Please provide:
1. **Browser console errors** (F12 → Console)
2. **Network errors** (F12 → Network → filter: failed)
3. **What you see** (blank page? error message? partial content?)

Then I can provide targeted fixes.
