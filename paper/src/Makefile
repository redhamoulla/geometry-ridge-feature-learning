PDF=conditional_geometric_register_of_observations_v7_en.pdf

all: $(PDF)

$(PDF): main.tex references.bib
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C
