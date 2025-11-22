/**
 * Simple test version of App
 * Use this to verify React is working
 */

function AppTest() {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="bg-white p-12 rounded-lg shadow-2xl max-w-2xl">
        <h1 className="text-4xl font-bold text-mlpurple mb-4">
          ✅ React is Working!
        </h1>
        <p className="text-gray-700 mb-4">
          If you see this message, the React app is loading correctly.
        </p>
        <div className="bg-mllavender3 p-4 rounded">
          <p className="text-sm text-mlpurple">
            <strong>Next step:</strong> Check browser console (F12) for any errors,
            then switch back to full Presentation component.
          </p>
        </div>
        <div className="mt-6 text-sm text-gray-600">
          <p>Server: http://localhost:5178/</p>
          <p>Status: Running</p>
          <p>Tailwind: {typeof document !== 'undefined' ? '✅ Loaded' : '❓'}</p>
        </div>
      </div>
    </div>
  );
}

export default AppTest;
