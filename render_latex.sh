#!/bin/bash
set -e

cd latex/experiment_1

# Run twice, LaTeX needs a second pass to resolve cross-references and
# the table of contents (first pass writes main.toc/main.aux, second
# pass reads them back in).
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Rendered latex/experiment_1/main.pdf"
