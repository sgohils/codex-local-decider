# Reproducing the research artifacts

This is an inconclusive five-day randomized pilot, not evidence of token savings.
The field window was October 4-9, 2026, ending at 5:27:21 p.m. America/New_York.
One baseline chat and no assisted chats enrolled. Final transcript truncation
invalidated all five baseline token measurements. Unknown values are `null`/NA.

The main PDF is generated with ReportLab from the same manuscript blocks as the
Markdown and editable standalone LaTeX. The PDF is not an export of a LaTeX
compiler. The LaTeX contains its own references and plot coordinates, requiring
no companion figure files. `delivery-manifest.json` records verification.

## Verify sanitized results

```text
python verify_aggregate_results.py aggregate-results.json
```

This checks coverage, missing values, the absence of per-chat/project identifiers,
and the 5.858% synthetic context calculation using `synthetic-results.json`.
The CSV has 100 generation-leader observations: 50 per model. These are not
monotone champion histories. Synthetic repetitions are not independent tasks.

## Regenerate from local private records

Use implementation commit `a777f2329bcf56769c208fd2a75d84de69c12755` in the private
`sgohils/codex-local-decider` repository. The real frozen study data and private
analysis are intentionally not distributed. With those local inputs, NumPy,
ReportLab, and the original font/runtime environment:

```text
python build_paper.py --data-dir LOCAL_STUDY --repo-dir DECIDER_CHECKOUT --output-dir PAPER_OUTPUT
```

`LOCAL_STUDY` contains `frozen.json`, `analysis-private.json`, `study.json`, and
private session summaries. Run the ordinary study analyzer to regenerate the
private analysis before building. Neither scripts nor paper regenerate hosted
coding trials. Keep actual prompts, transcript files, evidence, and task rows local.

Figures are provided as SVG/PDF/PNG. PNGs are rendered from the vector figure PDFs
with Poppler. The main PDF is visually checked using rendered page images. The
package includes all final paper files, aggregate metrics, references, source
builders, and synthetic plot data. Fonts and renderer versions can affect layout.

The normal candidate activation gate remains unmet. The experimental hook entries
were retained as no-ops under the expired, frozen registry using `--keep-hooks`.
Do not restart the study or activate the candidate on the strength of this pilot.
