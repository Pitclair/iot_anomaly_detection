# Thesis workspace

This is a separate academic writing workspace. The detector and full project
report are complete; dissertation preparation is listed in
[future work](../docs/future-work.md).

Build from any directory using the path to `build.sh`:

```sh
./thesis/build.sh
```

From inside this folder, run `./build.sh`. Docker uses the existing
`../compose.latex.yaml` service with LuaLaTeX and latexmk. The PDF is written to
`../out/thesis/thesis.pdf`; auxiliary files stay in that ignored output folder.

- `metadata.tex`: approved title, author, student ID, supervisors, and dates.
- `thesis.tex`: document order; remove inputs for optional items here.
- `frontmatter/`: one file per front-matter item, including separate lists.
- `chapters/`: one file per chapter; Chapters 1--2 contain the first prose draft.
- `backmatter/` and `references.bib`: bibliography output and cited sources.
- `appendices/`: reproducibility details outside the main-text page budget.
- `preamble.tex` and `thesis.xmpdata`: layout and PDF metadata.

Use the [writing guide](../docs/thesis_writing_guide.md) as the reference.
The agreed main text is 60–100 pages, with a target of 80–100; front matter,
bibliography, and appendices are excluded. The English abstract is based on the
official project abstract, revised to match the implementation and evidence.
The introduction and background/related-work chapters have a first prose draft,
with the cited subset of the [verified reference shortlist](../docs/thesis_reference_review.tex)
in `references.bib`. Chapters 3--7, the reproducibility appendix, and the Italian
summary remain a skeleton. Replace all bracketed placeholders before submission.

The first installment was drafted on 24 October 2026 against project revision
`63ba80d`. It covers the current joint DM/volume/ARP detector and the completed
final-partition evaluation's scope, without importing older reports' unfinished
work as current status. The current full project report and retained experiment
artifacts provide the technical evidence; the writing guide also contains
historical project-closure instructions. The chapter text, attribution, and
contribution wording require the author's review as drafting continues.

The container lacks Times New Roman. Drafts use TeX Gyre Termes, a bundled
Times-style font; Times New Roman is selected automatically if available.
The PDF/A settings do not replace validation of the final document.

## Template attribution

Adapted from [Template tesi UniPD DEI](https://www.overleaf.com/latex/templates/template-tesi-unipd-dei/dqwqrnbyyssj)
by **Cecco di Ronchitti da Bruzene**, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`upstream/main.tex` is the unmodified public source downloaded on 15 October
2026. The complete Overleaf project ZIP requires an account; the public source
is retained here instead. Changes include English text, separate files, the
chapter skeleton, font fallback, and the local Docker build.

`assets/unipd-seal.png` was extracted unchanged from the
[official English frontispiece](https://stem.elearning.unipd.it/pluginfile.php/562/mod_book/chapter/46/facsimile_frontespizio_tesi_cdl_%20nuovo%20ENG.docx).
University branding remains the University's property.
