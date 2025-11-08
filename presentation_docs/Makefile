## Makefile for LaTeX reports in presentation_docs

.PHONY: all assets results tech view-results view-tech clean cleanall help check-data

PYTHON ?= python
ROOT := $(abspath ..)
REQS := $(ROOT)/outputs/panels/hourly_panel.parquet \
        $(ROOT)/outputs/roll_signals/hourly_widening.csv \
        $(ROOT)/metadata/contracts_metadata.csv

all: assets results tech

# Generate figures/tables used by the reports
assets:
	@echo "Preparing analysis outputs (panel, widening, spreads)..."
	$(PYTHON) $(ROOT)/scripts/presentation_docs/prepare_report_assets.py --settings $(ROOT)/config/settings.yaml || exit 1
	@echo "Verifying required inputs..."
	$(MAKE) check-data
	@echo "Generating exploratory figures into outputs/exploratory ..."
	$(PYTHON) $(ROOT)/scripts/term_structure_plots.py || true
	$(PYTHON) $(ROOT)/scripts/term_structure_detailed_analysis.py || true
	$(PYTHON) $(ROOT)/scripts/volume_crossover.py || true
	$(PYTHON) $(ROOT)/scripts/contract_month_analysis.py || true
	@echo "Assets generation complete."

# Verify required inputs exist before building assets
check-data:
	@for f in $(REQS); do \
	  if [ ! -f $$f ]; then \
	    echo "Missing required input: $$f"; \
	    echo "Tip: run your analysis pipeline to refresh outputs/panels and outputs/roll_signals."; \
	    exit 1; \
	  fi; \
	done

# Compile Analytical Results report
results: analytical_results_report.pdf

analytical_results_report.pdf: analytical_results_report.tex
	@echo "Compiling Analytical Results report..."
	pdflatex -interaction=nonstopmode analytical_results_report.tex >/dev/null
	pdflatex -interaction=nonstopmode analytical_results_report.tex >/dev/null
	@echo "Analytical Results report ready."

# Compile Technical Implementation report
tech: technical_implementation_report.pdf

technical_implementation_report.pdf: technical_implementation_report.tex
	@echo "Compiling Technical Implementation report..."
	pdflatex -interaction=nonstopmode technical_implementation_report.tex >/dev/null
	pdflatex -interaction=nonstopmode technical_implementation_report.tex >/dev/null
	@echo "Technical Implementation report ready."

# View helpers (Linux/macOS)
view-results: analytical_results_report.pdf
	xdg-open analytical_results_report.pdf 2>/dev/null || open analytical_results_report.pdf

view-tech: technical_implementation_report.pdf
	xdg-open technical_implementation_report.pdf 2>/dev/null || open technical_implementation_report.pdf

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls *.synctex.gz
	@echo "Clean complete. PDF files preserved."

# Clean everything including PDFs
cleanall: clean
	@echo "Removing PDF files..."
	rm -f *.pdf
	@echo "All generated files removed."

# Help
help:
	@echo "Available targets:"
	@echo "  make (all)            - Generate assets and compile both PDFs"
	@echo "  make assets           - Regenerate figures under outputs/exploratory"
	@echo "  make results          - Build analytical_results_report.pdf"
	@echo "  make tech             - Build technical_implementation_report.pdf"
	@echo "  make view-results     - Open the Analytical Results PDF"
	@echo "  make view-tech        - Open the Technical Implementation PDF"
	@echo "  make clean|cleanall   - Remove auxiliary (and PDF) artifacts"
