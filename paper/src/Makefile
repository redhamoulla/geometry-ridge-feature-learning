BIBTEX ?= $(shell if command -v bibtex.original >/dev/null 2>&1; then echo bibtex.original; else echo bibtex; fi)

.PHONY: pdf clean

pdf:
	pdflatex -interaction=nonstopmode main.tex
	$(BIBTEX) main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot main.pdf
