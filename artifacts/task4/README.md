# Task 4 model outputs

Four files are **tracked in git**, and together they are the whole visual-search system:

| file | size | what it is |
|---|---:|---|
| `submitted/arcface_best.pt` | 45 MB | the ArcFace ResNet-18 encoder the hold-out benchmark chose |
| `submitted/gallery_embeddings.npy` | 70 MB | the catalogue, 33,968 x 512, already encoded |
| `submitted/gallery_ids.npy` | 0.3 MB | the catalogue ids, row-aligned with the embeddings |
| `submitted/image_preprocessing.json` | 1 KB | the letterbox size and channel statistics inference must reproduce |

The gallery is here on purpose. **Retrieval ranks a query against a catalogue, so the
encoder alone answers nothing** -- without these embeddings a clone could load the model
and still not return a single similar item. `query_embeddings.npy` and `query_ids.npy` are
the 3,774 hold-out queries, so the benchmark ranking reproduces without re-encoding
anything.

```console
python src/task4/retrieve_topk.py --from-saved-queries --top-k 10          # no images needed
python src/task4/retrieve_topk.py --images datasets/test/images_test       # encodes the test set
```

One thing to know about the gallery. The training run also left a gallery beside its
checkpoint under `models/arcface/`, and that one is an earlier development population of
30,389 items -- 3,579 short. The tracked pair in `submitted/` is the 33,968 the hold-out benchmark
actually scored, and every script and notebook reads it.

Everything else here is gitignored and lives on the team Drive: the four losing candidates
(CAE, Triplet, SupCon, Multi-Similarity) with their embeddings, the benchmark CSVs,
training history and figures.
