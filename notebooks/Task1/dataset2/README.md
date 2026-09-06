# Dataset2: external cosmetics training candidates

## Review status (audited 2026-09-05; documentation checked 2026-09-06)

**Not ready for direct integration.** Visual screening of all 699 images flagged
148 candidates (21.2%): 105 questionable labels and 43 poor crops. These are
provisional AI-assisted review decisions, not 148 confirmed annotation errors.
No images, labels, or quarantine contents were modified.

| Assigned class | Images | Label-review candidates | Crop-review candidates |
|---|---:|---:|---:|
| Compact | 100 | 0 | 0 |
| Concealer | 100 | 3 | 6 |
| Foundation and Primer | 100 | 1 | 1 |
| Highlighter and Blush | 100 | 2 | 19 |
| Kajal and Eyeliner | 99 | 34 | 7 |
| Lip Gloss | 100 | 14 | 5 |
| Lip Liner | 100 | 51 | 5 |
| Total | 699 | 105 | 43 |

The candidate sets do not overlap. The other 551 images had no obvious issue in
the visual screen; that is not independent confirmation of correct labels.

See [audit/visual_review.csv](audit/visual_review.csv) for every image's status,
and [the shared audit report](../../../docs/EXTERNAL_DATA_AUDIT.md) for methodology,
examples, limitations, and Task 1 integration implications. This register
preserves the earlier dataset2 screening; the subsequent full technical and
visual audit focused on [dataset1](../dataset1/README.md).

Examples requiring annotation review:

- [910117](images/910117.jpg): labelled Concealer, appears to show a Nivea cream container.
- [910271](images/910271.jpg): labelled Foundation and Primer, appears to show a moisturizer.
- [910512](images/910512.jpg) and [910549](images/910549.jpg): labelled Lip Gloss,
  appear to show nail-polish bottles.
- [910616](images/910616.jpg): labelled Lip Liner, appears to show exposed lipstick.

Examples of poor crops include [910132](images/910132.jpg), patterned fabric
labelled Concealer; [910280](images/910280.jpg), pale fabric labelled Foundation
and Primer; and [910306](images/910306.jpg), blue fabric labelled Highlighter
and Blush. An unusual background alone is not a failure: the issue is missing,
unrecognizable, or severely truncated target content.

## Files and labels

- `images/`: 699 readable JPEGs, all 60x80, IDs 910000 through 910699 except 910405.
- `external_cosmetics2.csv`: original supplied labels; unchanged.
- `audit/visual_review.csv`: full visual screening register and proposed actions.
- `images_gate_manifest.json` and `images_leakage_report.csv`: historical
  leakage reports supplied with the collection, using paths from another machine.
- `quarantine/`: the previously excluded 910405 crop and its original explanation.

The CSV columns are:

`id,gender,masterCategory,subCategory,articleType,season,usage,source,label_source`

Source is `external_cosmetics2_v1`. Gender, season, and usage were described as
propagated from the provided catalogue per articleType; they are not independent
source annotations. Every row is Women / Spring / Casual. Do not automatically
treat them as verified extra supervision for Tasks 2 and 3.

## Historical provenance and leakage records

The supplied documentation attributes this collection to Roboflow Universe's
"makeup products detection", reports CC BY 4.0, and describes COCO-object crops
from 2,076 photographs with 4,754 annotations. These remain inherited claims:
the exact upstream project/version URL, original archive, image/annotation IDs,
mapping rules, and preparation script are absent from this folder.

The old commands referenced preparation/verification scripts and paths that are
not present in this repository. The source annotations are needed to establish
whether a suspicious example resulted from an incorrect label, incorrect crop,
or occlusion. Do not automatically replace its label from visual inspection.

Historical leakage manifests report zero train/test matches and PASS. Their
`clean_and_usable=699` field is not a visual-quality certification. It does not
override the 148 review flags. The quarantine README attributes 910405 to a
dHash collision that was excluded conservatively; that historical investigation
has not been independently reproduced here.

## Task 1 use after curation

Keep accepted external rows in training only, appended after freezing the
provided-data split. The current supplied-only Task 1 split has 30,278 training
and 7,568 validation rows. The previous README's 27,596-row training denominator,
29,495-row combined total, and macro-F1 ceiling figures describe another setup
and should not be reused for this repository.

Dataset2 remains a training candidate requiring curation and is currently used by
[02_independent_evaluation.ipynb](../02_independent_evaluation.ipynb) as a robustness
probe for the supplied-only ensemble. Dataset1 has prepared enriched training manifests,
but the current training loader uses neither collection. Held-out status is model-specific:
exclude any collection used to train the model being evaluated. No `ExternalEval/`
directory is supplied here. See [evaluation scope](../../../docs/INDEPENDENT_EVALUATION_DATA.md).
Additional support can help rare classes but does not prove improved accuracy.
Compare against a supplied-only baseline using the same validation rows.

Use the original catalogue taxonomy, review the label mapping and crop quality,
and preserve image/label provenance. Version the accepted manifests and pixels
across machines before regenerating Task 1 worker notebooks. No integration
or training experiment was performed by this audit.
