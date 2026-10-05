# Build the technical reference: make -C docs/tex  (writes docs/g17-technical-reference.pdf)
PDF := ../g17-technical-reference.pdf

# The revision stamp on the cover and in the colophon: the commit the sources were built from, marked when
# docs/tex has uncommitted changes. Rewritten only when it changes, so an unchanged tree does not rebuild.
REV   := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unversioned)
DATE  := $(shell git log -1 --format=%cs 2>/dev/null)
DIRTY := $(shell git status --porcelain -- . 2>/dev/null | grep -v '\.pdf$$' | grep -q . && echo ' (with uncommitted changes)')

$(PDF): g17-technical-reference.tex preamble.tex mm-anchors.tex mm-aliases.tex opnames.tex values.tex opindex.tex revision.tex $(wildcard ch/*.tex) $(wildcard fig/*.tex)
	latexmk -lualatex -interaction=nonstopmode -halt-on-error g17-technical-reference.tex
	cp g17-technical-reference.pdf $(PDF)

# Opcode names, generated from the glossary (tools/g17docs.py --write-opnames); committed, and refreshed here.
opnames.tex: ../../isa/g17-opcode-glossary.json ../../tools/g17docs.py
	python3 ../../tools/g17docs.py --write-opnames

# Numbers quoted from generated artifacts (tools/g17docs.py --write-values); committed, and refreshed here.
values.tex: ../../isa/g17-coverage.json ../../isa/g17-agx3-renumber.json ../../isa/g17-opcode-glossary.json ../../tools/g17docs.py
	python3 ../../tools/g17docs.py --write-values

# Machine-model anchors, generated from its headings (not committed: a reworded heading must not break a citation).
mm-anchors.tex: ../g17-tensorops-machine-model.md ../../tools/g17docs.py
	python3 ../../tools/g17docs.py --write-mm-anchors

# The opcode index (appendix C), generated from the glossary and from the chapters' opcode mentions. Not
# committed: it follows every prose edit, so a committed copy would always be one edit behind.
opindex.tex: ../../isa/g17-opcode-glossary.json mm-anchors.tex ../../tools/g17docs.py $(wildcard ch/*.tex) $(wildcard fig/*.tex)
	python3 ../../tools/g17docs.py --write-opindex

revision.tex: FORCE
	@printf '\\newcommand{\\docrevision}{%s%s}\n\\newcommand{\\docrevdate}{%s}\n' '$(REV)' '$(DIRTY)' '$(DATE)' > revision.tmp
	@cmp -s revision.tmp revision.tex && rm -f revision.tmp || mv revision.tmp revision.tex

clean:
	latexmk -C g17-technical-reference.tex
	rm -f revision.tex revision.tmp opindex.tex mm-anchors.tex

.PHONY: clean FORCE
