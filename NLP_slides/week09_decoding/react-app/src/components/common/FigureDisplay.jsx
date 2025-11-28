/**
 * Figure Display Component
 * Handles both PNG images and PDF files with proper sizing
 */

import { Document, Page, pdfjs } from 'react-pdf';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const FigureDisplay = ({
  src,
  width = 0.75,  // fraction of textwidth (0.0-1.0)
  alt = "",
  className = "",
  centered = true
}) => {
  const isPDF = src?.endsWith('.pdf');
  const basePath = '/figures/';
  const fullPath = src?.startsWith('/') ? src : basePath + src;

  // Convert LaTeX textwidth fraction to pixels (assuming 1000px container)
  const pixelWidth = Math.round(1000 * width);

  const containerClass = centered
    ? 'flex justify-center items-center'
    : '';

  if (isPDF) {
    return (
      <div className={`figure-display ${containerClass} ${className}`}>
        <div style={{ width: `${width * 100}%` }}>
          <Document file={fullPath}>
            <Page
              pageNumber={1}
              width={pixelWidth}
              renderTextLayer={false}
              renderAnnotationLayer={false}
            />
          </Document>
        </div>
      </div>
    );
  }

  // PNG/JPG image
  return (
    <div className={`figure-display ${containerClass} ${className}`}>
      <img
        src={fullPath}
        alt={alt}
        style={{ width: `${width * 100}%` }}
        className="max-w-full h-auto"
      />
    </div>
  );
};

export default FigureDisplay;
