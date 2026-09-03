#!/bin/bash
set -e

ROOT="$(pwd)"

# Run twice, LaTeX needs a second pass to resolve cross-references and
# the table of contents (first pass writes .toc/.aux, second pass reads
# them back in).

cd "$ROOT/latex/experiment_1"
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "Rendered latex/experiment_1/main.pdf"

cd "$ROOT/latex/summary"
pdflatex -interaction=nonstopmode summary.tex
pdflatex -interaction=nonstopmode summary.tex
echo "Rendered latex/summary/summary.pdf"
