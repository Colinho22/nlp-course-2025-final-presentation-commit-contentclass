import './Madrid.css';

const MadridFooter = ({ current, total, title = "Week 9: Decoding Strategies", author = "NLP Course 2025" }) => {
  return (
    <footer className="madrid-footer bg-mllavender text-mlpurple px-6 py-2 text-xs border-t border-mllavender2">
      <div className="flex justify-between items-center">
        <div className="font-medium">{author}</div>
        <div className="font-semibold">{title}</div>
        <div className="font-mono font-bold">
          {current} / {total}
        </div>
      </div>
    </footer>
  );
};

export default MadridFooter;
