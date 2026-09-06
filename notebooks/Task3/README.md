# Task 3 — `gender` + `usage`

`03_task3_gender_usage_nguyen.ipynb` is the notebook. `task3_build.py` is the same
code as a plain script (`# %%` cell markers) — edit whichever you prefer; regenerate
the notebook from the script with the converter at the bottom of this file.

## What it answers

The brief says predict **both** `gender` and `usage` and does not say whether that is
one model or two. The notebook decides that with evidence:

| | design | backbones | outputs |
|---|---|---|---|
| **A** | two independent models | 2 | 5 · 8 |
| **B** | one model, joint label | 1 | 24 `gender × usage` pairs that occur |
| **C** | one shared backbone, two heads | 1 | 5 + 8 |

Same convolutional body, same seed, same schedule, same rows — so the difference
between them is the design choice and nothing else.

Then three ablations, one variable each: class-weighted loss, external data, and a
split that mimics the real test set.

## Running it on Colab

Setup is built into the notebook — sections 0.1 and 0.2 mount Drive, copy the zip,
unpack it to local disk and check the unpack finished. Nothing to add by hand.

**Before you start**, both of these must be reachable from *My Drive*. A shortcut
pointing into a Shared Drive is fine — Colab mounts My Drive only, so the shortcut is
the bridge. Anything under *Shared with me* is **not** mounted; right-click it in
Drive → **Add shortcut to Drive** first.

| | what | why |
|---|---|---|
| `ColabDataset.zip` | the provided catalogue, zipped | 43,577 separate Drive reads take ~30 min *per session*; one zip takes ~1 min |
| `A2_ExternalData/` | the collected images (folder, 20 MB) | small enough to read straight from Drive |

Then:

1. **Runtime → Change runtime type → T4 GPU.** It runs on CPU, but many times slower.
2. Upload the notebook (`File → Upload notebook`) or open it from Drive.
3. If your zip has a different name or location, edit `COLAB_ZIP` in section 0.1.
4. **Set `QUICK = True`** in section 0, then **Runtime → Run all**.

`QUICK` runs the whole notebook on 4,000 rows and 6 epochs in a few minutes. Its
*numbers are meaningless* — the point is to surface a typo before you spend an hour.
You are looking for one line at the very bottom:

```
saved -> task3_results.csv
```

5. Got that? Set `QUICK = False`, then **Runtime → Restart session and run all**.

### Expected cost

Decoding 37,745 JPEGs takes ~10 minutes and is cached to `/content/train_images.npy`,
so a re-run inside the same session skips it. Eight training runs of 20 epochs on a
T4 is roughly 30–45 minutes.

### If something goes wrong

| symptom | cause |
|---|---|
| `FileNotFoundError` listing your My Drive contents | `COLAB_ZIP` name is wrong — pick from the list it printed |
| `images missing -- the unzip was incomplete` | delete `/content/preprocessed_datasets` and re-run section 0.1 |
| external data "not found", §7 and §8.2 skipped | `A2_ExternalData` has no shortcut in My Drive |
| a cell hangs with no output | almost always an `unzip` overwrite prompt; section 0.1 passes `-o` to avoid it |

## Reading the output

Everything is **macro-F1**. Accuracy appears only next to it as evidence of why it
cannot be the metric: predicting `Casual` for every row scores **76.7%** accuracy on
`usage` and **0.109** macro-F1.

The three numbers worth carrying into the report:

* `usage` macro-F1 has a **hard ceiling of 0.500** if the four classes under 100
  images stay unlearnable (`Home` has **1** training image).
* the **forward split** (§8.1) is a better estimate of the graded score than the
  random one, because the test set is the highest ids and the label distribution
  drifts along that axis — `Women` goes 33% → 53%.
* the **independent evaluation** (§8.2) is the brief's §3.3 requirement, and the gap
  between it and validation is the finding, not a disappointment.

## Regenerating the notebook from the script

```bash
python -c "
import json
from pathlib import Path
src = Path('task3_build.py').read_text(encoding='utf-8')
cells, cur, kind = [], [], 'code'
def flush():
    global cur
    body = '\n'.join(cur).strip('\n')
    if body.strip():
        if kind == 'markdown':
            txt = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
            cells.append({'cell_type':'markdown','metadata':{},'source':txt.split('\n')})
        else:
            cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':body.split('\n')})
    cur = []
for line in src.split('\n'):
    if line.startswith('# %%'):
        flush(); kind = 'markdown' if '[markdown]' in line else 'code'; continue
    cur.append(line)
flush()
for c in cells: c['source'] = [l+'\n' for l in c['source'][:-1]] + [c['source'][-1]]
Path('03_task3_gender_usage_nguyen.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.0'},'colab':{'provenance':[],'toc_visible':True}},'nbformat':4,'nbformat_minor':5}, indent=1, ensure_ascii=False), encoding='utf-8')
print(len(cells), 'cells')
"
```
