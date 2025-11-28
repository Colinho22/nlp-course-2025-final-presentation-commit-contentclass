/**
 * TitleSlide Layout
 * Recreates Beamer [plain] frame with beamercolorbox
 * Pattern: Centered purple gradient box with shadow
 * Used by: Slide 1 (main title), Slide 43 (appendix divider)
 */

const TitleSlide = ({ title, subtitle, date, appendixSections }) => {
  return (
    <div className="slide-frame-plain h-full flex items-center justify-center bg-white">
      <div className="title-beamercolorbox bg-gradient-to-br from-mlpurple via-mllavender to-mllavender2 rounded-2xl shadow-2xl px-16 py-12 text-center max-w-4xl">
        {/* Main Title - \Huge */}
        <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
          {title}
        </h1>

        {/* Subtitle - \Large */}
        {subtitle && (
          <h2 className="text-3xl text-white opacity-95 mb-4">
            {subtitle}
          </h2>
        )}

        {/* Date - \normalsize */}
        {date && (
          <p className="text-xl text-white opacity-90 mt-6">
            {date}
          </p>
        )}

        {/* Appendix sections list (for Slide 43) */}
        {appendixSections && (
          <div className="mt-8 text-left">
            <p className="text-2xl text-white opacity-95 mb-4">
              25 slides: Complete mathematical treatment
            </p>
            <div className="space-y-2 mt-6">
              {appendixSections.map((section, i) => (
                <p key={i} className="text-lg text-white opacity-90">
                  <span className="font-bold">{section.range}:</span> {section.title}
                </p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TitleSlide;
