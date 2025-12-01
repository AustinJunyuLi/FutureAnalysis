## Makefile for LaTeX reports in presentation_docs

.PHONY: all report assets results tech seasonality view view-results view-tech legacy-reports clean cleanall help check-data

PYTHON ?= python
ROOT := $(abspath ..)
REQS := $(ROOT)/outputs/panels/hourly_panel.parquet \
        $(ROOT)/outputs/roll_signals/hourly_widening.csv \
        $(ROOT)/metadata/contracts_metadata.csv

# Primary target: unified report
all: report

# Build the unified comprehensive report
report: copper_futures_roll_analysis.pdf

copper_futures_roll_analysis.pdf: copper_futures_roll_analysis.tex
	@echo "Compiling unified Copper Futures Roll Analysis report..."
	pdflatex -interaction=nonstopmode copper_futures_roll_analysis.tex >/dev/null
	pdflatex -interaction=nonstopmode copper_futures_roll_analysis.tex >/dev/null
	@echo "Unified report ready: copper_futures_roll_analysis.pdf"

# View the unified report
view: copper_futures_roll_analysis.pdf
	xdg-open copper_futures_roll_analysis.pdf 2>/dev/null || open copper_futures_roll_analysis.pdf

# Legacy targets (deprecated - use 'report' instead)
legacy-reports: assets results tech seasonality

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

# Compile Technical Implementation report (legacy)
tech: technical_implementation_report.pdf

technical_implementation_report.pdf: technical_implementation_report.tex
	@echo "Compiling Technical Implementation report..."
	pdflatex -interaction=nonstopmode technical_implementation_report.tex >/dev/null
	pdflatex -interaction=nonstopmode technical_implementation_report.tex >/dev/null
	@echo "Technical Implementation report ready."

# Compile Seasonality report (legacy)
seasonality: seasonality_report.pdf

seasonality_report.pdf: seasonality_report.tex
	@echo "Compiling Seasonality report..."
	pdflatex -interaction=nonstopmode seasonality_report.tex >/dev/null
	pdflatex -interaction=nonstopmode seasonality_report.tex >/dev/null
	@echo "Seasonality report ready."

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
	@echo ""
	@echo "  PRIMARY:"
	@echo "  make (all)            - Build the unified copper_futures_roll_analysis.pdf"
	@echo "  make report           - Build the unified copper_futures_roll_analysis.pdf"
	@echo "  make view             - Open the unified report PDF"
	@echo ""
	@echo "  LEGACY (deprecated):"
	@echo "  make legacy-reports   - Build all legacy reports (results, tech, seasonality)"
	@echo "  make assets           - Regenerate figures under outputs/exploratory"
	@echo "  make results          - Build analytical_results_report.pdf"
	@echo "  make tech             - Build technical_implementation_report.pdf"
	@echo "  make seasonality      - Build seasonality_report.pdf"
	@echo "  make view-results     - Open the Analytical Results PDF"
	@echo "  make view-tech        - Open the Technical Implementation PDF"
	@echo ""
	@echo "  CLEANUP:"
	@echo "  make clean            - Remove auxiliary files (preserve PDFs)"
	@echo "  make cleanall         - Remove all generated files including PDFs"
