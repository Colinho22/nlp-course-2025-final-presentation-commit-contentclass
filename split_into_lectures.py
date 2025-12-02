"""
Split the Final Lecture into 4 standalone lectures.

Lecture 1: RAG (~28 slides)
Lecture 2: Agents (~20 slides) - needs new content
Lecture 3: Reasoning (~22 slides)
Lecture 4: Alignment (~21 slides)
"""
import re
from pathlib import Path
from datetime import datetime

# Source file
SOURCE_FILE = Path('presentations/20251129_2220_final_lecture.tex')

# Output directory
OUTPUT_DIR = Path('presentations')

# Timestamp for new files
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M')

# Lecture definitions with line ranges (approximate - will be refined)
LECTURES = {
    1: {
        'title': 'Retrieval-Augmented Generation',
        'subtitle': 'Grounding LLMs in External Knowledge',
        'filename': f'{TIMESTAMP}_lecture1_rag.tex',
        'sections': ['RAG'],  # Keywords to match
        'start_markers': ['HALLUCINATION PROBLEM', 'RAG SECTION'],
        'end_markers': ['AGENTS SECTION', 'BEYOND QA'],
    },
    2: {
        'title': 'AI Agents',
        'subtitle': 'Building LLMs That Take Action and Use Tools',
        'filename': f'{TIMESTAMP}_lecture2_agents.tex',
        'sections': ['Agents'],
        'start_markers': ['AGENTS SECTION', 'BEYOND QA'],
        'end_markers': ['SECTION: REASONING', 'COT SECTION'],
    },
    3: {
        'title': 'LLM Reasoning',
        'subtitle': 'From Chain-of-Thought to Test-Time Compute',
        'filename': f'{TIMESTAMP}_lecture3_reasoning.tex',
        'sections': ['Reasoning'],
        'start_markers': ['SECTION: REASONING', 'COT SECTION', 'COT DISCOVERY'],
        'end_markers': ['SECTION: ALIGNMENT', 'RLHF SECTION'],
    },
    4: {
        'title': 'AI Alignment',
        'subtitle': 'RLHF, DPO, and Making LLMs Safe',
        'filename': f'{TIMESTAMP}_lecture4_alignment.tex',
        'sections': ['Alignment'],
        'start_markers': ['SECTION: ALIGNMENT', 'RLHF SECTION', 'MISSING INGREDIENT'],
        'end_markers': ['\\end{document}'],
    },
}

# Preamble template (extracted from original)
PREAMBLE = r'''\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{multicol}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}

% Color definitions
\definecolor{mlblue}{RGB}{0,102,204}
\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender2}{RGB}{193,193,232}
\definecolor{mllavender3}{RGB}{204,204,235}
\definecolor{mllavender4}{RGB}{214,214,239}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}
\definecolor{mlred}{RGB}{214, 39, 40}
\definecolor{mlgray}{RGB}{127, 127, 127}
\definecolor{lightgray}{RGB}{240, 240, 240}
\definecolor{midgray}{RGB}{180, 180, 180}

% Apply custom colors to Madrid theme
\setbeamercolor{palette primary}{bg=mllavender3,fg=mlpurple}
\setbeamercolor{palette secondary}{bg=mllavender2,fg=mlpurple}
\setbeamercolor{palette tertiary}{bg=mllavender,fg=white}
\setbeamercolor{palette quaternary}{bg=mlpurple,fg=white}
\setbeamercolor{structure}{fg=mlpurple}
\setbeamercolor{section in toc}{fg=mlpurple}
\setbeamercolor{subsection in toc}{fg=mlblue}
\setbeamercolor{title}{fg=mlpurple}
\setbeamercolor{frametitle}{fg=mlpurple,bg=mllavender3}
\setbeamercolor{block title}{bg=mllavender2,fg=mlpurple}
\setbeamercolor{block body}{bg=mllavender4,fg=black}

% Remove navigation symbols
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize items}[circle]
\setbeamertemplate{enumerate items}[default]
\setbeamersize{text margin left=5mm,text margin right=5mm}

% Command for bottom annotation
\newcommand{\bottomnote}[1]{%
\vfill
\vspace{-2mm}
\textcolor{mllavender2}{\rule{\textwidth}{0.4pt}}
\vspace{1mm}
\footnotesize
\textbf{#1}
}

'''


def create_title_slide(lecture_num, title, subtitle):
    """Create a standalone title slide for a lecture."""
    return rf'''
% ==================== TITLE SLIDE ====================
\begin{{frame}}[plain]
\vspace{{1cm}}
\begin{{center}}
{{\Huge {title}}}\\[0.3cm]
{{\Large {subtitle}}}\\[1cm]
{{\normalsize NLP Course -- Lecture {lecture_num}}}\\[0.3cm]
{{\small Advanced Topics in Natural Language Processing}}
\end{{center}}
\end{{frame}}
'''


def create_intro_slides(lecture_num, title):
    """Create intro slides for context."""
    intros = {
        1: r'''
% ==================== LECTURE INTRO ====================
\begin{frame}[t]{Why Do LLMs Hallucinate?}
\begin{columns}[T]
\column{0.48\textwidth}
\textbf{The Problem}
\begin{itemize}
\item LLMs predict the most likely next token
\item They have no access to real-time information
\item Knowledge is frozen at training time
\item No mechanism to verify facts
\end{itemize}

\column{0.48\textwidth}
\textbf{The Solution: RAG}
\begin{itemize}
\item Retrieve relevant documents first
\item Augment the prompt with facts
\item Generate grounded responses
\item Cite sources for verification
\end{itemize}
\end{columns}

\vspace{0.5cm}
\begin{center}
\textcolor{mlpurple}{\textbf{This lecture: How to ground LLMs in external knowledge}}
\end{center}
\bottomnote{RAG is the most widely deployed technique for making LLMs factually accurate.}
\end{frame}
''',
        2: r'''
% ==================== LECTURE INTRO ====================
\begin{frame}[t]{What If LLMs Could DO Things?}
\begin{columns}[T]
\column{0.48\textwidth}
\textbf{LLMs Today}
\begin{itemize}
\item Excellent at generating text
\item Answer questions from training data
\item No ability to take actions
\item Cannot interact with the world
\end{itemize}

\column{0.48\textwidth}
\textbf{LLM Agents}
\begin{itemize}
\item Use tools (search, code, APIs)
\item Execute multi-step plans
\item Observe results and adapt
\item Accomplish real-world tasks
\end{itemize}
\end{columns}

\vspace{0.5cm}
\begin{center}
\textcolor{mlpurple}{\textbf{This lecture: Building AI systems that take action}}
\end{center}
\bottomnote{Agents transform LLMs from passive responders to active problem solvers.}
\end{frame}
''',
        3: r'''
% ==================== LECTURE INTRO ====================
\begin{frame}[t]{Why Do LLMs Struggle with Reasoning?}
\begin{columns}[T]
\column{0.48\textwidth}
\textbf{The Problem}
\begin{itemize}
\item LLMs give instant responses
\item No ``working memory'' for computation
\item Multi-step problems require planning
\item Direct answers often wrong
\end{itemize}

\column{0.48\textwidth}
\textbf{The Solution}
\begin{itemize}
\item Chain-of-Thought: think step by step
\item Test-time compute: think longer
\item Process reward models: verify steps
\item DeepSeek-R1: trained to reason
\end{itemize}
\end{columns}

\vspace{0.5cm}
\begin{center}
\textcolor{mlpurple}{\textbf{This lecture: Teaching LLMs to think before answering}}
\end{center}
\bottomnote{Reasoning capabilities have dramatically improved since 2022.}
\end{frame}
''',
        4: r'''
% ==================== LECTURE INTRO ====================
\begin{frame}[t]{The Alignment Problem}
\begin{columns}[T]
\column{0.48\textwidth}
\textbf{Raw Pre-trained LLMs}
\begin{itemize}
\item Not helpful (ignore instructions)
\item Not honest (confidently wrong)
\item Not harmless (generate toxic content)
\item Just predict likely tokens
\end{itemize}

\column{0.48\textwidth}
\textbf{Aligned LLMs}
\begin{itemize}
\item Follow user instructions
\item Refuse harmful requests
\item Admit uncertainty
\item Helpful, Honest, Harmless
\end{itemize}
\end{columns}

\vspace{0.5cm}
\begin{center}
\textcolor{mlpurple}{\textbf{This lecture: How to align AI with human values}}
\end{center}
\bottomnote{Alignment is what transforms GPT-3 into ChatGPT.}
\end{frame}
''',
    }
    return intros.get(lecture_num, '')


def create_closing_slides(lecture_num, title):
    """Create closing slides with takeaways."""
    closings = {
        1: r'''
% ==================== KEY TAKEAWAYS ====================
\begin{frame}[t]{Key Takeaways: RAG}
\begin{enumerate}
\item \textbf{RAG solves hallucination} by grounding LLMs in external documents
\item \textbf{Vector search} enables millisecond retrieval from billions of documents
\item \textbf{HNSW} provides O(log n) approximate nearest neighbor search
\item \textbf{Chunking strategy} critically affects retrieval quality
\item \textbf{RAG can fail} at retrieval, ranking, or generation stages
\end{enumerate}

\vspace{0.5cm}
\textbf{Key Equations:}
\begin{itemize}
\item Dense retrieval: $\text{sim}(q, d) = \cos(E_q(q), E_d(d))$
\item RAG probability: $p(y|x) = \sum_z p(z|x) \cdot p(y|x,z)$
\end{itemize}
\bottomnote{RAG is the foundation of most production LLM applications today.}
\end{frame}

% ==================== RESOURCES ====================
\begin{frame}[t]{Further Reading: RAG}
\textbf{Foundational Papers:}
\begin{itemize}
\item Lewis et al. (2020) - ``Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks''
\item Karpukhin et al. (2020) - ``Dense Passage Retrieval''
\item Malkov \& Yashunin (2018) - ``HNSW: Hierarchical Navigable Small World Graphs''
\end{itemize}

\vspace{0.3cm}
\textbf{Tools \& Frameworks:}
\begin{itemize}
\item Vector DBs: Pinecone, Weaviate, ChromaDB, FAISS
\item Frameworks: LangChain, LlamaIndex
\end{itemize}
\bottomnote{Repository: github.com/Digital-AI-Finance/Natural-Language-Processing}
\end{frame}
''',
        2: r'''
% ==================== KEY TAKEAWAYS ====================
\begin{frame}[t]{Key Takeaways: AI Agents}
\begin{enumerate}
\item \textbf{Agents extend LLMs} from text generators to action takers
\item \textbf{ReAct pattern}: Think $\to$ Act $\to$ Observe $\to$ Repeat
\item \textbf{Tool use} enables interaction with external systems
\item \textbf{MCP} standardizes how agents connect to tools
\item \textbf{Memory} is critical for maintaining context across actions
\item \textbf{Evaluation} of agents is challenging but essential
\end{enumerate}

\vspace{0.3cm}
\textbf{Key Insight:} Agents are still unreliable for complex tasks -- human oversight remains essential.
\bottomnote{Agent capabilities are rapidly improving but require careful deployment.}
\end{frame}

% ==================== RESOURCES ====================
\begin{frame}[t]{Further Reading: AI Agents}
\textbf{Foundational Papers:}
\begin{itemize}
\item Yao et al. (2023) - ``ReAct: Synergizing Reasoning and Acting''
\item Schick et al. (2023) - ``Toolformer''
\item Significant-Gravitas - AutoGPT
\end{itemize}

\vspace{0.3cm}
\textbf{Frameworks \& Tools:}
\begin{itemize}
\item LangChain, LangGraph, CrewAI
\item Claude Code, Cursor, Devin
\item Model Context Protocol (MCP)
\end{itemize}
\bottomnote{Repository: github.com/Digital-AI-Finance/Natural-Language-Processing}
\end{frame}
''',
        3: r'''
% ==================== KEY TAKEAWAYS ====================
\begin{frame}[t]{Key Takeaways: LLM Reasoning}
\begin{enumerate}
\item \textbf{Chain-of-Thought} dramatically improves reasoning (+40\% on math)
\item \textbf{Intermediate tokens} serve as computational scratchpad
\item \textbf{Test-time compute} is the new scaling paradigm
\item \textbf{DeepSeek-R1} showed pure RL can develop reasoning
\item \textbf{Process reward models} enable verification of reasoning steps
\end{enumerate}

\vspace{0.3cm}
\textbf{Key Insight:} ``Let the model think longer'' is often more effective than making models bigger.
\bottomnote{Reasoning capabilities define the frontier of AI capabilities in 2025.}
\end{frame}

% ==================== RESOURCES ====================
\begin{frame}[t]{Further Reading: LLM Reasoning}
\textbf{Foundational Papers:}
\begin{itemize}
\item Wei et al. (2022) - ``Chain-of-Thought Prompting''
\item Wang et al. (2023) - ``Self-Consistency''
\item DeepSeek (2025) - ``DeepSeek-R1''
\item OpenAI (2024) - ``o1 System Card''
\end{itemize}

\vspace{0.3cm}
\textbf{Key Concepts:}
\begin{itemize}
\item Test-time compute scaling
\item Process Reward Models (PRMs)
\item GRPO (Group Relative Policy Optimization)
\end{itemize}
\bottomnote{Repository: github.com/Digital-AI-Finance/Natural-Language-Processing}
\end{frame}
''',
        4: r'''
% ==================== KEY TAKEAWAYS ====================
\begin{frame}[t]{Key Takeaways: AI Alignment}
\begin{enumerate}
\item \textbf{RLHF} transforms base LLMs into helpful assistants
\item \textbf{Reward models} learn human preferences from comparisons
\item \textbf{PPO + KL penalty} prevents reward hacking
\item \textbf{DPO} simplifies alignment (no separate reward model)
\item \textbf{Constitutional AI} enables self-improvement with principles
\end{enumerate}

\vspace{0.3cm}
\textbf{Open Questions:}
\begin{itemize}
\item Whose values should AI systems align with?
\item How do we align AI smarter than humans?
\end{itemize}
\bottomnote{Alignment is what makes AI systems safe and beneficial.}
\end{frame}

% ==================== FINAL MESSAGE ====================
\begin{frame}[plain]
\begin{center}
\vspace{2cm}
{\Huge The Convergence}\\[1cm]
{\Large \textcolor{mlblue}{USEFUL} + \textcolor{mlorange}{SMART} + \textcolor{mlgreen}{SAFE}}\\[0.5cm]
{\large RAG \& Agents + Reasoning + Alignment}\\[1.5cm]
{\normalsize ``The models predict tokens.\\[0.2cm]
\textbf{You} decide what we build with them.''}
\end{center}
\end{frame}

% ==================== QUESTIONS ====================
\begin{frame}[plain]
\begin{center}
\vspace{3cm}
{\Huge Questions?}\\[1cm]
{\large Thank you for your attention}\\[0.5cm]
{\small github.com/Digital-AI-Finance/Natural-Language-Processing}
\end{center}
\end{frame}
''',
    }
    return closings.get(lecture_num, '')


def extract_frames(content, start_markers, end_markers):
    """Extract frames between start and end markers."""
    # Find start position
    start_pos = 0
    for marker in start_markers:
        pos = content.find(marker)
        if pos != -1:
            # Find the beginning of the frame before this marker
            frame_start = content.rfind('\\begin{frame}', 0, pos)
            if frame_start != -1:
                start_pos = frame_start
                break
            else:
                start_pos = pos
                break

    # Find end position
    end_pos = len(content)
    for marker in end_markers:
        pos = content.find(marker, start_pos)
        if pos != -1:
            # Find the end of the previous frame
            frame_end = content.rfind('\\end{frame}', start_pos, pos)
            if frame_end != -1:
                end_pos = frame_end + len('\\end{frame}')
                break
            else:
                end_pos = pos
                break

    return content[start_pos:end_pos]


def remove_quantlet_branding(content):
    """Remove Quantlet branding tikzpicture blocks."""
    # Pattern to match Quantlet branding blocks
    pattern = r'\s*% Quantlet branding \(auto-generated\).*?\\end\{tikzpicture\}'
    return re.sub(pattern, '', content, flags=re.DOTALL)


def count_frames(content):
    """Count number of frames in content."""
    return len(re.findall(r'\\begin\{frame\}', content))


def create_lecture(lecture_num, lecture_info, source_content):
    """Create a complete lecture file."""
    title = lecture_info['title']
    subtitle = lecture_info['subtitle']

    # Extract frames
    frames = extract_frames(
        source_content,
        lecture_info['start_markers'],
        lecture_info['end_markers']
    )

    # Remove Quantlet branding (will be re-added later)
    frames = remove_quantlet_branding(frames)

    # Build complete document
    doc = PREAMBLE
    doc += f"\\title{{{title}}}\n"
    doc += f"\\subtitle{{{subtitle}}}\n"
    doc += "\\author{NLP Course}\n"
    doc += "\\institute{MSc Program}\n"
    doc += "\\date{\\today}\n\n"
    doc += "\\begin{document}\n"

    # Title slide
    doc += create_title_slide(lecture_num, title, subtitle)

    # Intro slides
    doc += create_intro_slides(lecture_num, title)

    # Main content
    doc += "\n% ==================== MAIN CONTENT ====================\n"
    doc += frames

    # Closing slides
    doc += "\n% ==================== CLOSING ====================\n"
    doc += create_closing_slides(lecture_num, title)

    doc += "\n\\end{document}\n"

    return doc


def main():
    """Main execution."""
    print("Splitting Final Lecture into 4 standalone lectures...")
    print("=" * 60)

    # Read source file
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_content = f.read()

    print(f"Source file: {SOURCE_FILE}")
    print(f"Total frames in source: {count_frames(source_content)}")
    print()

    # Create each lecture
    for num, info in LECTURES.items():
        print(f"Creating Lecture {num}: {info['title']}")

        lecture_content = create_lecture(num, info, source_content)
        frame_count = count_frames(lecture_content)

        # Save file
        output_path = OUTPUT_DIR / info['filename']
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(lecture_content)

        print(f"  -> {output_path}")
        print(f"  -> {frame_count} frames")
        print()

    print("=" * 60)
    print("COMPLETE: Created 4 lecture files")
    print()
    print("Next steps:")
    print("1. Review and adjust content in each lecture")
    print("2. Add new slides for Lecture 2 (Agents)")
    print("3. Compile each lecture with pdflatex")
    print("4. Run add_all_branding.py for Quantlet branding")


if __name__ == '__main__':
    main()
