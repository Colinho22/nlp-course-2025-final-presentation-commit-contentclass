# Makefile for Embeddings Modular Presentation
# Cross-platform build system

# Configuration
MAIN = main
OUTPUT_DIR = build
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -output-directory=$(OUTPUT_DIR)
BIBTEX = bibtex
VIEWER = start  # Windows default, change for Linux/Mac

# File lists
TEX_FILES := $(wildcard *.tex) $(wildcard content/*.tex) $(wildcard appendix/*.tex) $(wildcard styles/*.tex)
FIGURE_DIRS := ../figures/embeddings ../charts_3d

# Output files
PDF = $(OUTPUT_DIR)/$(MAIN).pdf
AUX_FILES = $(OUTPUT_DIR)/*.aux $(OUTPUT_DIR)/*.log $(OUTPUT_DIR)/*.nav \
            $(OUTPUT_DIR)/*.out $(OUTPUT_DIR)/*.snm $(OUTPUT_DIR)/*.toc \
            $(OUTPUT_DIR)/*.vrb $(OUTPUT_DIR)/*.bbl $(OUTPUT_DIR)/*.blg

# Phony targets
.PHONY: all clean view quick full handout presentation web help variants watch

# Default target
all: full

# Create output directory
$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

# Quick build (single pass)
quick: $(OUTPUT_DIR)
	@echo "Quick build (single pass)..."
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN) $(MAIN).tex
	@echo "Build complete: $(PDF)"

# Full build (multiple passes for references)
full: $(OUTPUT_DIR)
	@echo "Full build (3 passes)..."
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN) $(MAIN).tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN) $(MAIN).tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN) $(MAIN).tex
	@cp $(PDF) $(MAIN).pdf
	@echo "Build complete: $(MAIN).pdf"

# Handout version (multiple slides per page)
handout: $(OUTPUT_DIR)
	@echo "Building handout version..."
	@echo "\\PassOptionsToClass{handout}{beamer}" > $(OUTPUT_DIR)/handout_config.tex
	@cat $(OUTPUT_DIR)/handout_config.tex $(MAIN).tex > $(OUTPUT_DIR)/$(MAIN)_handout.tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN)_handout $(OUTPUT_DIR)/$(MAIN)_handout.tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN)_handout $(OUTPUT_DIR)/$(MAIN)_handout.tex
	@cp $(OUTPUT_DIR)/$(MAIN)_handout.pdf .
	@echo "Handout complete: $(MAIN)_handout.pdf"

# Presentation mode with notes
notes: $(OUTPUT_DIR)
	@echo "Building presentation with notes..."
	@echo "\\setbeameroption{show notes on second screen}" > $(OUTPUT_DIR)/notes_config.tex
	@cat $(OUTPUT_DIR)/notes_config.tex $(MAIN).tex > $(OUTPUT_DIR)/$(MAIN)_notes.tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN)_notes $(OUTPUT_DIR)/$(MAIN)_notes.tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN)_notes $(OUTPUT_DIR)/$(MAIN)_notes.tex
	@cp $(OUTPUT_DIR)/$(MAIN)_notes.pdf .
	@echo "Notes version complete: $(MAIN)_notes.pdf"

# Print version (no overlays)
print: $(OUTPUT_DIR)
	@echo "Building print version..."
	@echo "\\PassOptionsToClass{handout}{beamer}" > $(OUTPUT_DIR)/print_config.tex
	@echo "\\setbeamertemplate{background canvas}{}" >> $(OUTPUT_DIR)/print_config.tex
	@cat $(OUTPUT_DIR)/print_config.tex $(MAIN).tex > $(OUTPUT_DIR)/$(MAIN)_print.tex
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(MAIN)_print $(OUTPUT_DIR)/$(MAIN)_print.tex
	@cp $(OUTPUT_DIR)/$(MAIN)_print.pdf .
	@echo "Print version complete: $(MAIN)_print.pdf"

# Build all variants
variants: full handout notes print
	@echo "All variants built successfully!"

# Web version (requires tex4ht)
web: $(OUTPUT_DIR)
	@echo "Building HTML version..."
	@htlatex $(MAIN).tex "xhtml,charset=utf-8" " -cunihtf -utf8"
	@mkdir -p $(OUTPUT_DIR)/html
	@mv *.html *.css $(OUTPUT_DIR)/html/ 2>/dev/null || true
	@echo "Web version complete: $(OUTPUT_DIR)/html/$(MAIN).html"

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -f $(AUX_FILES)
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.bbl *.blg
	@rm -f *_handout.pdf *_notes.pdf *_print.pdf
	@echo "Clean complete"

# Deep clean (including output)
distclean: clean
	@echo "Deep cleaning..."
	@rm -rf $(OUTPUT_DIR)
	@rm -f $(MAIN).pdf
	@echo "Distribution clean complete"

# View the PDF
view: full
	@echo "Opening PDF..."
	@$(VIEWER) $(MAIN).pdf &

# Watch for changes and rebuild (requires inotify-tools on Linux)
watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -qre close_write *.tex content/*.tex appendix/*.tex styles/*.tex; \
		make quick; \
		echo "Rebuilt at $$(date)"; \
	done

# Validate structure
validate:
	@echo "Validating LaTeX structure..."
	@echo "Checking for undefined references..."
	@grep -n "undefined" $(OUTPUT_DIR)/*.log || echo "No undefined references"
	@echo "Checking for missing figures..."
	@grep -n "File.*not found" $(OUTPUT_DIR)/*.log || echo "No missing figures"
	@echo "Checking for overfull boxes..."
	@grep -c "Overfull" $(OUTPUT_DIR)/*.log || echo "No overfull boxes"
	@echo "Validation complete"

# Generate dependency graph
dependencies:
	@echo "Generating dependency graph..."
	@echo "digraph G {" > dependencies.dot
	@echo "  $(MAIN) -> preamble;" >> dependencies.dot
	@echo "  $(MAIN) -> metadata;" >> dependencies.dot
	@for file in content/*.tex; do \
		echo "  $(MAIN) -> \"$$file\";" >> dependencies.dot; \
	done
	@for file in appendix/*.tex; do \
		echo "  $(MAIN) -> \"$$file\";" >> dependencies.dot; \
	done
	@echo "}" >> dependencies.dot
	@dot -Tpdf dependencies.dot -o dependencies.pdf
	@echo "Dependency graph: dependencies.pdf"

# Count statistics
stats:
	@echo "Presentation Statistics:"
	@echo "========================"
	@echo -n "Total slides: "
	@grep -c "\\\\begin{frame}" $(TEX_FILES) | awk -F: '{sum+=$$2} END {print sum}'
	@echo -n "Total lines: "
	@wc -l $(TEX_FILES) | tail -1 | awk '{print $$1}'
	@echo -n "Figures: "
	@grep -c "includegraphics" $(TEX_FILES) | awk -F: '{sum+=$$2} END {print sum}'
	@echo -n "Equations: "
	@grep -c "begin{equation\\|\\$\\$" $(TEX_FILES) | awk -F: '{sum+=$$2} END {print sum}'
	@echo -n "Files: "
	@ls -1 $(TEX_FILES) | wc -l

# Help target
help:
	@echo "Embeddings Presentation Makefile"
	@echo "================================"
	@echo "Available targets:"
	@echo "  make quick      - Single-pass compilation (fast)"
	@echo "  make full       - Full compilation (3 passes)"
	@echo "  make handout    - Create handout version"
	@echo "  make notes      - Create presenter notes version"
	@echo "  make print      - Create print-friendly version"
	@echo "  make variants   - Build all variants"
	@echo "  make web        - Generate HTML version"
	@echo "  make view       - Build and open PDF"
	@echo "  make clean      - Remove auxiliary files"
	@echo "  make distclean  - Remove all generated files"
	@echo "  make watch      - Auto-rebuild on changes"
	@echo "  make validate   - Check for errors"
	@echo "  make stats      - Show presentation statistics"
	@echo "  make help       - Show this help"

# Platform-specific adjustments
ifeq ($(OS),Windows_NT)
	VIEWER = start
	RM = del /Q
	MKDIR = mkdir
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		VIEWER = open
	else
		VIEWER = xdg-open
	endif
	RM = rm -f
	MKDIR = mkdir -p
endif