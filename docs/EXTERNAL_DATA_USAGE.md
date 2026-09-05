# Using the external data — one line per task

`src/external_data.py`. Companion to `src/preprocessing.py`: that module owns the
provided catalogue, this one owns everything collected from outside it.

## Setup (Colab)

Download **`A2_ExternalData`** from the team Drive into your own Drive, then:

```python
from google.colab import drive
drive.mount('/content/drive')
```

That is the whole setup. `A2_ExternalData` is one of the paths the loader searches,
so nothing else is needed. If you put it somewhere else:

```python
import os
os.environ['A2_EXTERNAL_DATA'] = '/content/drive/MyDrive/<folder containing ExternalCosmetics>'
```

Locally: put the three folders inside `Dataset/` next to `datasets/`.

Check it is visible before you train:

```python
import sys; sys.path.append('../src')
import external_data as ed
ed.summary()
```

## Adding it to training — Tasks 1, 2, 3

**Two lines.** Split first, then add:

```python
import preprocessing as pp
import external_data as ed

frame = pp.load_manifest(target=TARGET)
training, validation = pp.make_split(frame, TARGET)

training = ed.add_to_training(training, TARGET)     # <-- the only new line
```

`add_to_training` takes the **training** frame from `make_split`, never the whole
manifest. That is deliberate and it is the point of the whole design:

> External rows must reach `train` only. If they were appended before `make_split`,
> some would land in validation, validation would stop being purely provided
> catalogue imagery, and every "with vs without external data" comparison would
> become meaningless — the exact thing this data was collected to demonstrate.

The signature enforces it, so it cannot be got wrong by forgetting.

## What you get

Every external row is `Women` / `Spring` / `Casual`. That single fact lands
completely differently on each target, so **this data is not "on" or "off" for the
whole project — it is a per-target decision**:

| Task | target | rows | where the label lands | imbalance max/min | verdict |
|---|---|---|---|---|---|
| 1 | `articleType` | 1,899 | 10 starved cosmetics classes, all 100% inside the test-like id range | macro-F1 ceiling 0.742 → **0.782** | **use** |
| 2 | `season` | 1,899 | `Spring`, the **rarest** season (4.0% of train) | 13× → **5×** | **use** |
| 3 | `gender` | 1,899 | `Women`, already the 2nd largest (36.6%) | 31× → 31× | pointless — no gain |
| 3 | `usage` | 1,899 | `Casual`, already the **majority** (77.2% → 78.6%) | 21,256× → **23,155×** | **harmful — leave it out** |
| 4 | — | not used | retrieval runs on the catalogue | — | n/a |

`usage` is the one to be careful about. Its whole difficulty is a long tail, and
adding 1,899 rows of the majority class makes that tail *relatively rarer*. The
loader prints a warning when this happens rather than letting it pass silently:

```
!! WARNING: this made 'usage' MORE imbalanced, not less: 21,256x -> 23,155x
   Every external row is usage='Casual', which is already the majority class.
```

A measured negative result is worth reporting — it is evidence of judgement, which
is the criterion actually being marked. So run `usage` both ways and report the
comparison rather than quietly dropping the data.

## The one caveat you must report — Task 2 especially

Every cosmetics class is `season=Spring` in the provided data (100% of all 12
classes; `Personal Care` overall is 99.6% Spring). So these labels are propagated,
not observed, and every row carries `label_source` saying so.

That triples Spring, which is a real gain **and** a real bias in the same move.
Spring was already 65% `Personal Care`; afterwards it is far higher. The model is
being taught harder that a cosmetic implies Spring — which is the actual rule
generating the label, but it is not seasonality.

**Before claiming Task 2 improved, measure Spring recall separately on Personal Care
and non-Personal-Care validation rows.** If only the former improves, the gain is a
categorical shortcut and the report has to say so. The Spring rows that are not
Personal Care — Casual Shoes 77, Sports Shoes 60, Tshirts 45 — get no help at all.

## The ablation — please run this, a marker will ask

Nobody can currently answer "how much did the external data help", because the best
Task 1 number changed architecture, class weighting **and** the data at the same
time. One extra run fixes it:

```python
USE_EXTERNAL = True     # flip to False for the second run, change nothing else
training, validation = pp.make_split(frame, TARGET)
if USE_EXTERNAL:
    training = ed.add_to_training(training, TARGET)
```

Same seed, same architecture, same everything else. The difference between the two
macro-F1 values **is** the contribution. Put both numbers in the report.

## The independent evaluation set — do not train on it

```python
evaluation = ed.load_eval_set()      # 261 images, 52 classes, all four targets
```

Separate function on purpose. These 261 in-the-wild photographs are the answer to
the brief's §3.3 *"data collected completely outside of the scope of your original
training and evaluation"*, and their entire value is that no model here has seen
them. Training on them once destroys that permanently.

Report each task's metric on this set **beside** the same metric on validation. The
gap between the two is the finding — it is how much of the model's skill was about
fashion items and how much was about Myntra's photography convention.

Known limits, for the report:

- `season` skews Summer (74%, against 49.6% in the provided data) because
  season-agnostic accessories were defaulted to the modal value. `season` here
  partly measures the annotator's prior. There are **no Spring images at all**.
- Conversely `usage=Home` has **4** images, against **1** in the entire provided
  training set — this set can measure classes the provided data cannot teach.
- All labels come from a single annotator. A second person labelling a random 50
  without seeing these gives an inter-annotator agreement figure, which is worth
  reporting in its own right.

## Provenance, if anyone asks

| Set | Images | Source | Licence | Leakage gate |
|---|---|---|---|---|
| `ExternalCosmetics` | 1,200 | "Cosmetic Images" COCO archive | **CC0** | PASS — 0 vs test, 0 vs train |
| `ExternalCosmetics2` | 699 | Roboflow "makeup products detection" | **CC BY 4.0** | PASS — 0 vs test, 0 vs train |
| `ExternalEval` | 261 | Openverse (Flickr, Wikimedia, NASA, …) | CC BY / CC0 / PDM | PASS — 0 vs test, 0 vs train |

Every folder has its own `README.md` with the full method, the measured domain gap,
and how to re-run the leakage check. `ExternalEval` also ships `ATTRIBUTION.txt`,
which must travel with the images because most of them are CC BY.

Those three READMEs are also committed here, as
`docs/external_sets/ExternalCosmetics.md`, `ExternalCosmetics2.md` and
`ExternalEval.md`, so the provenance survives even if the Drive folder does not.

Re-verify the images themselves with:

```bash
python src/verify_external_data.py --check <folder>/images     # leakage gate
python src/verify_external_data.py --domain-gap                # background difference
```

`--domain-gap` prints the measured difference between each external set's backgrounds
and the catalogue's, both as stored and after the 60×80 transform. Quote the **as
loaded** row in the report — `ExternalCosmetics` is stored at native crop size, so
white padding moves its border brightness from 101.5 to 201.2 against the catalogue's
247.2, and only the padded version is what a model sees.

And check the loader still behaves before you trust a training run:

```bash
python tests/test_external_data.py
```

11 checks, including "no external row can reach validation" and "the evaluation set
never appears in a training frame". It skips cleanly with instructions if the images
are not on the machine.
