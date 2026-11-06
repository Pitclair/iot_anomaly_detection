# Thesis workspace

The complete thesis draft is ready for the author's final review:
**64 pages of main text** in the UniPD modern template, plus front matter,
bibliography, and the reproducibility appendix, for 94 pages total. The compiled document is
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
- `thesis.tex`: template metadata and document order.
- `unipd-thesis-modern.cls` and `template/`: the imported UniPD class and branding.
- `docs/`: thin adapters from the template front matter to the thesis text.
- `upstream-template/`: the complete, unchanged 27-file Overleaf template project.
- `TEMPLATE_README.md`, `main.tex`, demo chapters, `img/`, and optional document
  files: convenient root-level copies of the original template material.
- `frontmatter/`: one file per front-matter item, including separate lists.
- `chapters/`: seven complete chapter drafts.
- `backmatter/` and `references.bib`: bibliography output and cited sources.
- `appendices/`: reproducibility details outside the main-text page budget.
- `preamble.tex` and `thesis.xmpdata`: project-specific packages and PDF metadata.
- `figures/`: vector diagrams and plots rendered by TikZ/PGFPlots.
- `generated/`: checked table rows, plot CSVs, and source fingerprints.
- `validation.json`: final document identity, environment, and verification.

Use the [writing guide](../docs/thesis_writing_guide.md) for formatting and the
institution's submission guidance. The agreed main text is 60–100 pages; the
modern-template draft remains within that range. The cited subset of the
[verified reference shortlist](../docs/thesis_reference_review.tex) is in
`references.bib`. The shortlist retains annotations explaining what each source
supports and where access limited verification. Mosimann's paper is used for
historical attribution; its derivation pages were not available in that review.

The draft preserves the distinctions needed for a defensible final review:

- The closing replay has pooled F1 80.00% on 1,008 unique windows, with
  substantially different device results and only 18 positive windows.
- Later labels had already been inspected in retrospective audits. The fixed
  closing replay records retrospective evidence.
- Conservative modes reject every closing update. Other-duration threshold
  adaptation can increase false alerts; learning gains require accepted updates
  and improved later outcomes.
- Numerical speed depends on the operation. D-Link is reserved for algorithm
  benchmarking; Chromecast and Samsung supply detection evidence.
- The constrained resource study receives supplied counts on an x86 Docker
  host. Physical gateways and live collection require separate measurements.

The appendix separates historical experiments from the current source and
the fresh verification. All 643 current project tests passed without skips,
including the available image-dependent checks. Raw preprocessing, all
historical fits, and long resource trials were not repeated for writing.

The author should review the prose, attribution, contribution wording, Italian
summary, and submission declarations. The cover preserves academic year
**2025/2026**, which still needs personal confirmation; an unconfirmed
graduation date is omitted. Optional personal acknowledgements are omitted.

The final PDF uses the template's **TeX Gyre Pagella X**, 11 pt, A4 layout,
page geometry, chapter styling, headers, and UniPD red link colour. It does not
depend on locally cached proprietary fonts.

The final document passes **veraPDF 1.30.3, PDF/A-2b** validation. The report is
[pdfa_validation.xml](../out/thesis/pdfa_validation.xml). Recheck after any
document edit with the locally installed validator:

```sh
out/thesis/tools/verapdf/verapdf --flavour 2b --format xml \
  out/thesis/thesis.pdf > out/thesis/pdfa_validation.xml
```

PDF/A-2 validation confirms the embedded fonts. Layout checks found no
overfull boxes or unresolved references or
citations. The cover, abstracts, figures, and representative tables were
visually inspected. The PDF is below the 40 MB submission limit.

## Template attribution

The document uses **unipd-thesis-modern v1.3.1** by Francesco Barone, released
under a CC licence and published through
[GitHub](https://github.com/baronefr/unipd-thesis-modern) and
[Overleaf](https://www.overleaf.com/latex/templates/unipd-modern-thesis-template/qxcgfbvhfjtn).
The class and its `template/` assets were copied unchanged on 2 November 2026
from the author's `Fatemeh_Master_Thesis` Overleaf project. Thesis-specific
wiring remains in `thesis.tex`, `preamble.tex`, and `docs/`; the scientific
chapters, evidence, figures, tables, and references were not rewritten during
the template migration. The original dedication is active, and every unused
template file is retained unchanged under `upstream-template/` even where an
active thesis file occupies the original path. University branding remains the
University's property.
