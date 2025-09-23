import os
import re
import glob

def extract_figure_references():
    """Extract all figure references from LaTeX files"""
    references = []
    
    # Get all .tex files
    tex_files = glob.glob("../*.tex") + glob.glob("../weeks/*.tex")
    
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all includegraphics references
            pattern = r'\\includegraphics\[.*?\]\{figures/(.*?)\}'
            matches = re.findall(pattern, content)
            
            for match in matches:
                references.append({
                    'file': os.path.basename(tex_file),
                    'figure': match
                })
                
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
    
    return references

def get_existing_figures():
    """Get list of existing figures"""
    figures_dir = "../figures"
    if not os.path.exists(figures_dir):
        return []
    
    return [f for f in os.listdir(figures_dir) if f.endswith('.pdf')]

def main():
    print("Checking for missing figures...")
    print("=" * 50)
    
    # Get referenced figures
    references = extract_figure_references()
    referenced_figures = set([ref['figure'] for ref in references])
    
    # Get existing figures
    existing_figures = set(get_existing_figures())
    
    # Find missing figures
    missing_figures = referenced_figures - existing_figures
    
    print(f"\nReferenced figures: {len(referenced_figures)}")
    print(f"Existing figures: {len(existing_figures)}")
    print(f"Missing figures: {len(missing_figures)}")
    
    if missing_figures:
        print("\nMISSING FIGURES:")
        print("-" * 30)
        for missing in sorted(missing_figures):
            print(f"  - {missing}")
            # Show which files reference this missing figure
            refs = [ref['file'] for ref in references if ref['figure'] == missing]
            print(f"    Referenced in: {', '.join(set(refs))}")
            print()
    
    # Also check for unused figures
    unused_figures = existing_figures - referenced_figures
    if unused_figures:
        print("\nUNUSED FIGURES (exist but not referenced):")
        print("-" * 45)
        for unused in sorted(unused_figures):
            print(f"  - {unused}")
    
    print("\nAll referenced figures:")
    print("-" * 25)
    for fig in sorted(referenced_figures):
        status = "✓ EXISTS" if fig in existing_figures else "✗ MISSING"
        print(f"  {status}: {fig}")

if __name__ == "__main__":
    main()