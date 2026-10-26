# Thesis workspace

The complete thesis draft is ready for the author's final review:
**80 pages of main text**, plus front matter, bibliography, and the
reproducibility appendix. The compiled document is
[thesis.pdf](../out/thesis/thesis.pdf). All seven chapters, the English abstract,
and the Italian summary are written. There are 30 cited references, eight
figures, and an evidence map connecting results to retained experiment files.

Build from the repository root:

```sh
./thesis/build.sh
```

From inside this folder, run `./build.sh`. Other working directories can use
the script's absolute path. The build first regenerates tables and plot data
using the standard-library `generate_evidence.py`, then uses the existing
`../compose.latex.yaml` service with LuaLaTeX, Biber, and latexmk. The PDF and
auxiliary files stay in the ignored `../out/thesis/` directory. Docker uses the
locally available TeX Live image without network access.

Regeneration requires the retained processed windows, labels, configurations,
models, and experiment records in this checkout; raw PCAPs are not needed.
It checks confusion-count arithmetic, all 3,024 closing events against exact
label coverage, agreement of the three modes, and the 109 frozen dependencies.
Three later scoring source files are accepted only when their hashes match
the recorded regression. The generated manifest lists all 112 presentation
sources and both historical/current source boundaries. Historical experiments
are read, rather than rerun or overwritten.

- `metadata.tex`: title, author, student ID, supervisors, and academic year.
- `thesis.tex`: document order.
- `frontmatter/`: one file per front-matter item, including separate lists.
- `chapters/`: seven complete chapter drafts.
- `backmatter/` and `references.bib`: bibliography output and cited sources.
- `appendices/`: reproducibility details outside the main-text page budget.
- `preamble.tex` and `thesis.xmpdata`: layout and PDF metadata.
- `figures/`: vector diagrams and plots rendered by TikZ/PGFPlots.
- `generated/`: checked table rows, plot CSVs, and source fingerprints.
- `validation.json`: final document identity, environment, and verification.

Use the [writing guide](../docs/thesis_writing_guide.md) for formatting and the
institution's submission guidance. The agreed main text is 60–100 pages, with
a target of 80–100; the draft meets that target. The cited subset of the
[verified reference shortlist](../docs/thesis_reference_review.tex) is in
`references.bib`. The shortlist retains annotations explaining what each source
supports and where access limited verification. Mosimann's paper is used for
historical attribution; its derivation pages were not available in that review.

The draft preserves the distinctions needed for a defensible final review:

- The closing replay has pooled F1 80.00% on 1,008 unique windows, with
  substantially different device results and only 18 positive windows.
- Later labels had already been inspected in retrospective audits. The fixed
  closing replay is not described as an untouched external validation set.
- Conservative modes promote no closing updates. Other-duration threshold
  adaptation can increase false alerts; static equality is not a learning gain.
- Numerical speed depends on the operation. The unlabelled D-Link workload
  supports numerical comparisons, not attack precision or recall.
- The constrained resource study receives supplied counts on an x86 Docker
  host. Physical gateways and live collection were not measured.

The appendix separates historical experiments from the current source and
the fresh verification. All 643 current project tests passed without skips,
including the available image-dependent checks. Raw preprocessing, all
historical fits, and long resource trials were not repeated for writing.

The author should review the prose, attribution, contribution wording, Italian
summary, and submission declarations. The cover preserves academic year
**2025/2026**, which still needs personal confirmation; an unconfirmed
graduation date is omitted. Optional personal acknowledgements are omitted.

The final PDF uses **Times New Roman**, 12 pt, with the required A4 layout and
1.5 spacing. Original Core Fonts files are cached locally in the ignored
`out/thesis/fonts/` directory; their SHA-256 values match the
[Debian installer records](https://sources.debian.org/src/msttcorefonts/3.8.1/update-ms-fonts/).
The original archive and [font licence](https://corefonts.sourceforge.net/eula.htm)
are retained under `out/thesis/tools/corefonts/`, outside version control.
A fresh checkout needs these font files or an installed Times New Roman;
otherwise the build warns and uses TeX Gyre Termes.

The final document passes **veraPDF 1.30.3, PDF/A-2b** validation. The report is
[pdfa_validation.xml](../out/thesis/pdfa_validation.xml). Recheck after any
document edit with the locally installed validator:

```sh
out/thesis/tools/verapdf/verapdf --flavour 2b --format xml \
  out/thesis/thesis.pdf > out/thesis/pdfa_validation.xml
```

The preamble omits LuaTeX's optional CIDSet entry to avoid an incorrect
TrueType subset listing; PDF/A-2 validation confirms the resulting embedded
fonts. Layout checks found no overfull boxes or unresolved references or
citations. The cover, abstracts, figures, and representative tables were
visually inspected. The PDF is below the 40 MB submission limit.

## Template attribution

Adapted from [Template tesi UniPD DEI](https://www.overleaf.com/latex/templates/template-tesi-unipd-dei/dqwqrnbyyssj)
by **Cecco di Ronchitti da Bruzene**, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`upstream/main.tex` is the unmodified public source downloaded on 15 October
2026. The complete Overleaf project ZIP requires an account; the public source
is retained here instead. Changes include English text, separate files, the
modular chapters, statistical figures/tables, font handling, and local Docker build.

`assets/unipd-seal.png` was extracted unchanged from the
[official English frontispiece](https://stem.elearning.unipd.it/pluginfile.php/562/mod_book/chapter/46/facsimile_frontespizio_tesi_cdl_%20nuovo%20ENG.docx).
University branding remains the University's property.
