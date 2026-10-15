# Master's Thesis Writing Guide

Verified against the current University of Padua and DEI information on 6 October 2026.

The page-count guidance below was checked again on 15 October 2026.

## References that still matter for writing

The Uniweb application and graduation-deadline work is already complete. These are the remaining references relevant to producing the thesis itself:

1. [DEI master's-thesis instructions](https://stem.elearning.unipd.it/mod/book/view.php?chapterid=46&id=234&lang=en)

   This is the authoritative page for language, layout, frontispiece, final file, and supervisor approval.

2. [Official English master's-thesis frontispiece](https://stem.elearning.unipd.it/pluginfile.php/562/mod_book/chapter/46/facsimile_frontespizio_tesi_cdl_%20nuovo%20ENG.docx)

   Its appearance may be adapted, but all information in the official model must remain.

3. [DEI master's-degree grading criteria](https://stem.elearning.unipd.it/pluginfile.php/562/mod_book/chapter/46/criteri%20voto%20magistrale.pdf)

   The evaluation considers originality and results, accuracy and effort, clarity of the thesis and presentation, and the significance of the theoretical or experimental methodology.

4. [UniPD generative-AI guidelines for theses](https://wwwassets.unipd.it/sites/default/files/2026-05/Guidelines_Generative_AI_Tools_in_Teaching_Writing_Bachelor_Doctoral_Theses.pdf)

   AI may assist with outlining, source organisation, feedback, and language improvement, but it must not write substantial portions of the thesis. Its use must be declared transparently.

5. [UniPD bibliography and anti-plagiarism guidance](https://biblio.unipd.it/strumenti-di-ricerca/gestione-bibliografie)

   DEI does not currently specify one mandatory citation style. Confirm the choice with the supervisor; IEEE numeric style is a natural option for Computer Engineering.

6. Optional: [Community DEI thesis LaTeX template on Overleaf](https://www.overleaf.com/latex/templates/template-tesi-unipd-dei/dqwqrnbyyssj)

   Link verified on 15 October 2026. A local English adaptation is available in [`thesis/`](../thesis/README.md), with separate front-matter and chapter files and a tested LuaLaTeX build. Its `fontspec` setup requires XeLaTeX or LuaLaTeX. This community template is a starting point; check the layout and frontispiece against the official DEI instructions. Recent work in the [Padua Thesis Archive](https://thesis.unipd.it/) can be used to inspect realistic structure and presentation.

7. For the final build only: [Uniweb final-document upload guide](https://wwwassets.unipd.it/sites/default/files/2026-05/Linee%20Guida%20Upload%20documento%20definitivo%20la%20prova%20finale%2013052026.pdf) and [PDF/A instructions](https://wwwassets.unipd.it/sites/default/files/pdf_imported_files/Istruzioni-pdfa-accessibile.pdf).

## Formatting that applies

The current DEI instructions indicate:

- English language for Computer Engineering.
- Times New Roman, 12 pt.
- Line spacing 1.5.
- A4 pages.
- Top and bottom margins: 2 cm.
- Outside margin: 2 cm.
- Inside margin: 3 cm.
- Official English frontispiece as the first page.
- Final file in PDF/A, maximum 40 MB.
- Filename: `Surname_Name.pdf`.
- The title on the frontispiece must exactly match the title approved in Uniweb.
- The supervisor must approve the uploaded PDF/A.

The current DEI page does not specify a mandatory bibliography style. Confirm the choice with the supervisor. Page-count guidance is recorded separately below.

Use the abstract already submitted in Uniweb as a working summary. Revise the final thesis abstract only after the results and conclusions are stable.

## Thesis length: minimum and maximum pages

**Checked on 15 October 2026 for the Computer Engineering master's degree (Laurea Magistrale, DEI).**

| Question | Finding in the official sources checked |
|---|---|
| Minimum thesis pages | No numerical minimum found. |
| Maximum thesis pages | No numerical maximum found. |

The [DEI master's-thesis instructions](https://stem.elearning.unipd.it/mod/book/view.php?chapterid=46&id=234&lang=en), under **Prova finale (Lauree Magistrali)**, specify formatting and submission requirements without a page count. The **10–20 pages** mentioned earlier on that page concern the **bachelor's presentation**, not the master's thesis. The **40 MB** restriction concerns the final PDF/A file size, not its length.

The 2020 Computer Engineering programme framework in the [official LM-32 archive](https://wwwassets.unipd.it/sites/default/files/2026-06/LM32_Ingegneria%20Informatica%20-%20Computer%20engineering.zip), linked from the [University's Engineering regulations page](https://www.unipd.it/regolamento-didattico-scuola-ingegneria), also describes the final thesis without setting a numerical page count.

These findings concern the public sources checked; supervisor or cohort-specific instructions may add requirements.

### Agreed writing budget

- **Minimum:** 60 pages of main text.
- **Target:** 80–100 pages of main text.
- **Maximum:** 100 pages of main text.

Count from the Introduction through the Conclusions, including figures and tables. Exclude front matter, bibliography, and appendices.

The student confirmed the supervisor's agreement to this budget on 15 October 2026. These are the agreed project writing limits, separate from University requirements. Keep each chapter focused on the evidence needed to answer the research questions.

## Decisions to freeze with the supervisor

Before drafting chapters, obtain agreement on:

1. The exact thesis title and scope.
2. The research questions and claimed contributions.
3. Which experiments must be completed before submission.
4. Whether adaptation is a central contribution or only an experiment.
5. The desired citation style. The length budget above is already agreed with the supervisor.
6. Acceptance of the proposed chapter structure and LaTeX template.
7. How the use of AI should be disclosed.

Do not wait for every experiment to be complete before writing. Freeze the questions and experiment plan, then write the stable background and methodology while the final experiments run.

## Project-side wrap-up before final writing

The project is functional, but its evidence is spread across several branches and reports. Closing that gap is more important than adding another feature.

### 1. Choose one authoritative thesis revision

Record one branch and commit as the system described by the thesis. The current repository has separate work for:

- the reference detector and adaptation on `master`;
- LM documentation and LaTeX tooling on `feature/lm-loglikelihood`;
- rolling temporal evaluation on `feature/temporal-model`;
- wider traffic-signal experiments on `feature/volume-anomaly`.

Do not describe all branch features as one implemented system unless they are actually consolidated and rerun together. The smallest defensible choice is to use the reference detector plus the LM validation and one clearly identified adaptation study. Treat temporal and wider-signal branches as separate experiments or future work unless they are deliberately integrated.

### 2. Freeze the claims before freezing the code

For each research question, decide the exact evidence required:

| Research question | Minimum evidence |
|---|---|
| Static detection | Chronological labelled results for both selected devices, with confusion matrices and episode-level interpretation |
| LM backend | Numerical agreement, decision agreement, and runtime comparison against SciPy |
| Adaptation | Static versus selected adaptive policy, plus buffer-contamination and parameter-movement checks |

Anything not needed for one of those rows is optional. Forecasting, automated reaction, neural extensions, and production edge deployment should not delay the thesis.

### 3. Resolve the experimental protocol

Before the definitive runs, write down and stop changing:

- selected devices and datasets;
- traffic categories and window length;
- fit, calibration, development, and final periods;
- threshold rule;
- static and adaptive configurations;
- metrics and episode definition;
- random seeds;
- final-test access rule.

Development data may select a configuration. Final data may evaluate it once; it must not trigger another tuning cycle.

### 4. Produce one final evidence bundle

The final run should leave behind:

- the exact Git commit;
- configuration files;
- dataset fingerprints;
- model and threshold artifacts;
- event files;
- evaluation and benchmark reports;
- the commands needed to reproduce the run;
- the software and hardware environment;
- one machine-readable source for every number used in the thesis.

Do not manually copy numbers from old reports when a current JSON artifact can provide them.

### 5. Run the closure checks

Before calling the project finished:

1. Run the complete automated test suite on the authoritative revision.
2. Run the documented smoke workflow from raw or public fixture input to evaluation output.
3. Rerun the final experiment matrix from clean output paths.
4. Check that every table and plot agrees with its underlying artifact.
5. Confirm that results do not mix incompatible branches, datasets, thresholds, or model versions.
6. Record known failures and negative experiments instead of silently removing them.
7. Tag or otherwise record the final thesis revision so later edits cannot change the reported system unnoticed.

## Recommended thesis scope

The shortest defensible thesis from this project is:

> Design and evaluation of a reproducible Dirichlet–multinomial anomaly-detection pipeline for IoT traffic, including validation of the Languasco–Migliardi likelihood backend and an experimental comparison of static and guarded adaptive detection.

Possible research questions:

- **RQ1:** How effectively does a Dirichlet–multinomial model detect anomalous IoT traffic under chronological evaluation?
- **RQ2:** How does the native Languasco–Migliardi backend compare with SciPy in numerical agreement and runtime?
- **RQ3:** Can guarded adaptation improve detection without learning from attack traffic?

Do not expand the central thesis to forecasting, reaction selection, theoretical topic models, and full edge deployment unless the supervisor explicitly requires them. The current evidence does not yet support all those claims.

## Suggested structure

### Front matter

- Official frontispiece
- Abstract
- Optional acknowledgements
- Table of contents
- List of figures and tables
- Acronyms, only if genuinely useful

### Chapter 1 — Introduction

Explain:

- The IoT security problem.
- Why signature-based detection is insufficient.
- Why device-specific, unsupervised detection is relevant.
- Thesis objective and research questions.
- The specific contributions.
- A short chapter overview.

### Chapter 2 — Background and related work

Combine the theory and literature instead of creating several oversized chapters:

- IoT network anomaly detection.
- Count-vector representation of traffic.
- Multinomial and Dirichlet–multinomial models.
- Overdispersion.
- Parameter estimation.
- Yu–Shaw and Languasco–Migliardi numerical methods.
- Existing IoT applications and the gap addressed by this work.

Every mathematical definition and literature claim needs a primary citation.

### Chapter 3 — LM-IDNet methodology and implementation

Describe the final system, not its development history:

- PCAP validation and device isolation.
- Ten-minute aggregation and traffic categories.
- Temporal partitions.
- Model fitting.
- Independent threshold calibration.
- Scoring and anomaly decisions.
- LM native backend.
- Adaptation mechanism and safeguards.
- Artifact fingerprints and reproducibility.

Include one architecture figure and concise pseudocode. Do not paste implementation code.

### Chapter 4 — Experimental methodology

Define the experiment before presenting results:

- Datasets and their provenance.
- Devices, dates, windows, and labels.
- Fit, calibration, development, and final partitions.
- Leakage-prevention policy.
- Static and adaptive configurations.
- Baselines.
- Metrics: precision, recall, F1, false positives, episode detection, and runtime.
- Software and hardware environment.
- Random seeds, commit identifier, configuration files, and dataset fingerprints.

### Chapter 5 — Results

Present facts without interpretation first:

- Dataset statistics.
- Static-detector results.
- LM–SciPy numerical agreement.
- LM–SciPy performance.
- Adaptation comparison.
- Contamination and parameter-movement checks.
- Final chronological evaluation.

Use tables and plots generated directly from final artifacts.

### Chapter 6 — Discussion and limitations

Answer the research questions and explain:

- What worked.
- What failed.
- Why protocol proportions miss some attacks.
- Why adaptation can become contaminated.
- Whether LM's accuracy or speed advantage is workload-dependent.
- Dataset and device generalisability.
- Threats to validity.
- What cannot honestly be claimed.

### Chapter 7 — Conclusions and future work

Provide:

- One direct answer per research question.
- A short list of contributions.
- The principal limitation.
- Only realistic future work: additional signals, more devices, safer adaptation, and edge benchmarking.

## Technical and visual writing rules

### Mathematical notation

- Define every symbol at first use and keep one notation throughout the thesis.
- Use bold lowercase symbols for vectors, such as \(\boldsymbol{x}\) and \(\boldsymbol{\alpha}\), and ordinary italic symbols for scalars.
- Distinguish the Dirichlet concentration vector \(\boldsymbol{\alpha}\), total concentration \(\alpha_0\), and any dispersion quantity such as \(\psi\). State the exact parameterisation.
- State whether a reported likelihood is the complete Dirichlet–multinomial log-likelihood, a parameter-dependent fitting kernel, or a per-window anomaly score.
- Number only equations that are referenced later.
- Give units for every duration, window size, memory value, and runtime.

### Figures and tables

- Each figure or table must support a specific claim in the surrounding text.
- Captions must be understandable without searching the chapter for basic context.
- Label axes, include units, and state whether higher or lower values are better.
- Prefer vector PDF plots; use high-resolution raster images only when vector output is inappropriate.
- Use a colour-blind-safe palette and do not encode meaning through colour alone.
- Use consistent device names, metric precision, ordering, colours, and symbols across all figures.
- Avoid screenshots of terminals, JSON, source code, or tables that can be typeset directly.
- Do not use vertical rules in tables. Align comparable numeric columns and avoid meaningless decimal precision.

### Experimental reporting

- State the protocol before the result so the reader can judge the evidence independently.
- Separate development results from final-test results visibly.
- Report confusion-matrix counts with precision, recall, and F1; accuracy alone is misleading for imbalanced anomaly data.
- Report per-capture or per-episode behaviour where pooled window metrics hide temporal variation.
- Report negative results when they explain a design decision.
- Never describe a selected development result as unbiased final performance.
- Distinguish measured facts from interpretations and proposed explanations.

### Implementation reporting

- Explain design decisions and algorithms, not every class and function.
- Use one architecture diagram and short pseudocode where they make the method clearer.
- Identify which code was supplied, which code was adapted, and which parts are original contributions.
- Cite a fixed repository revision when discussing implementation details.
- Put long command sequences, schema listings, and secondary diagnostic tables in an appendix only if they materially support reproducibility.

### Citations and terminology

- Cite the original paper or dataset publication, not a local PDF filename or a secondary summary.
- Add citations while drafting; do not leave bibliography reconstruction until the end.
- Use one citation style consistently.
- Define acronyms once and use them only when they save repeated text.
- Use one stable term for each concept: for example, `window`, `capture`, `attack episode`, `calibration set`, and `final set` must not be interchangeable.
- Clearly mark reproduced or adapted figures and verify that their licences permit reuse. Prefer redrawing concepts and generating result plots from the project's own data.

### Prose

- Give each section one purpose and each paragraph one main claim.
- Prefer direct statements over descriptions of the writing process.
- Avoid claims such as "efficient", "robust", "lightweight", or "real-time" unless a reported measurement supports them.
- Use past tense for experiments performed, present tense for mathematical definitions and established facts, and future tense only for genuine future work.
- End each results subsection with the answer it provides, not with a repetition of every number.

## Writing order

Do not write from Chapter 1 to Chapter 7 in order.

1. Freeze the research questions, scope, and final experiment protocol with the supervisor.
2. Choose the authoritative branch and create a short project-closure checklist from the section above.
3. Create the correctly formatted LaTeX thesis shell and bibliography file.
4. Create an evidence table mapping every intended claim to a primary source, source-code component, experiment artifact, figure, or table.
5. Write Chapter 3 because the implemented method is already mostly known.
6. Write Chapter 4 and make its protocol the contract for the final runs.
7. Finish the minimum project work required by that protocol, run the tests, and produce the final evidence bundle.
8. Generate Chapter 5 directly from the frozen artifacts.
9. Write Chapter 6 while the results and failed experiments are fresh.
10. Write Chapter 2 using primary publications and consistent citations.
11. Write Chapter 1 after it is clear exactly what the thesis demonstrates.
12. Write Chapter 7, then revise the already-submitted abstract into the final thesis abstract.
13. Perform separate scientific, language, citation, visual, formatting, and PDF/A reviews.
14. Give the supervisor enough time for at least one complete review before producing the submitted version.

For individual sections, use the same small pattern:

> Claim → evidence or citation → interpretation → limitation.

## How to reuse this repository

Do not start from a blank page. The main sources are:

- [`docs/thesis_proposal_and_system_architecture.tex`](thesis_proposal_and_system_architecture.tex): thesis proposal and theoretical foundation.
- [`docs/lm_idnet_full_project_report.tex`](lm_idnet_full_project_report.tex): current full project report and evidence.
- [`docs/lm_idnet_project_status_report.tex`](lm_idnet_project_status_report.tex): current project status.
- [`docs/lm_idnet_adaptation_stage_report.tex`](lm_idnet_adaptation_stage_report.tex): adaptation design and results.
- [`docs/lm_backend_change_record.tex`](lm_backend_change_record.tex): native LM integration history and validation.

Treat these documents as notes and evidence, not text to copy. Rewrite everything in your own academic voice and verify every result and citation.

Use the prepared [`thesis/`](../thesis/README.md) workspace for the actual thesis. Run `./thesis/build.sh` from the repository root to generate `out/thesis/thesis.pdf` through the existing Compose service. Edit `thesis/metadata.tex` for the approved title and personal details, and write in the separate chapter and front-matter files.

The older reports use a different layout. The thesis workspace uses 12 pt text, 1.5 spacing, and the indicative DEI margins. It falls back to TeX Gyre Termes when Times New Roman is unavailable in the container. Its PDF/A settings still require validation of the final document before submission.

Most importantly, the full project report says that the untouched final evaluation, temporal consolidation, adaptation safety, and edge evaluation are incomplete. Freeze a smaller final claim and complete its evidence before writing the definitive Results and Abstract.

## Definition of done

The project and thesis are ready for final review when all of the following are true:

- One commit identifies the implemented system described in the thesis.
- The final protocol was written before the final evaluation.
- Every reported number is traceable to a retained artifact.
- The full tests and one end-to-end reproduction command pass.
- Results, limitations, and failed experiments are internally consistent.
- The thesis answers each research question without claiming unsupported production readiness.
- The bibliography contains every cited source and no uncited filler.
- The final PDF follows the DEI layout, uses the official frontispiece, and validates as PDF/A.
