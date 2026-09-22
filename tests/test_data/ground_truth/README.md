# Visual Ground Truth — PERT/CPM Diagram Corpus

Each file `<image stem>.json` in this folder is the **hand-annotated visual
ground truth** for one corpus image in `tests/test_data/Imag PERT/`.

> Determination rule: `tests/test_data/Imag PERT/<NAME>.<ext>` →
> `tests/test_data/ground_truth/<NAME>.json` (same stem, different folder).
> All 11 stems are unique, so the mapping is bijective and deterministic.

This folder is the reference for the accuracy benchmark
(`pert_analyzer.benchmark.accuracy`). It is **never** produced by the CV
pipeline; it is authored by a human looking at the pixels. Annotating with
pipeline output would make the benchmark circular and meaningless.

---

## Status of these files

- `1.json` is a **complete canonical gold standard** (verified against
  `tests/test_data/reference_diagrams/reference_aon_expected.json`: 22
  activities, 28 dependencies, CPM duration 54, 16 critical paths).
- The other 10 files are **drafts seeded from detector output** — ids,
  positions and a dependency guess are already filled, but a human must
  verify them against the image before they are trustworthy.

Until an entry is filled in (activities/events/dependencies), the annotation
is an honest `UNCERTAIN` placeholder and the accuracy report shows the image
as having no comparable reference items.

Legend for each annotation:

| Status | Meaning |
| --- | --- |
| `COMPLETE` | I can see every node/arrow and its geometry + durations. |
| `PARTIAL` | Some items are visible, some not (with `uncertain` listing what is excluded). |
| `UNCERTAIN` | I cannot reliably read this diagram (blurry/skewed/illegible) — do not guess. |

**Never fabricate**: anything you cannot read from the image is put under
`uncertain` (or the file stays `UNCERTAIN`). The benchmark excludes uncertain
items from precision/recall/F1 math.

---

## Schema (v1.0)

```jsonc
{
  "schema_version": "1.0",
  "image": "1.png",              // the corpus file name
  "diagram_type": "AON",         // "AON" | "AOA"
  "annotation_status": "UNCERTAIN", // COMPLETE | PARTIAL | UNCERTAIN
  "activities": [ ... ],         // AON: nodes ; AOA: arrows
  "events": [ ... ],             // AOA only
  "dependencies": [ ... ],       // explicit edges (optional if derived from activities)
  "expected_activities": 22,     // optional human count (informational)
  "expected_events": 13,         // optional human count
  "expected_dependencies": 28,   // optional human count
  "uncertain": {                 // optional: what to exclude from metrics
    "activities": ["X"], "events": [], "dependencies": [["A","B"]]
  },
  "notes": "…"
}
```

### AON activity (a node)

```jsonc
{
  "id": "A",                    // the label written on the node (or "N1" if unlabeled)
  "duration": 5.0,              // number inside/next to the node if present
  "position": [100.0, 100.0],   // centre of the node, image pixels, origin top-left
  "bounding_box": [80.0, 80.0, 40.0, 40.0], // [x, y, w, h]
  "predecessors": [],           // ids of nodes pointing into this one
  "successors": ["B"]           // ids of nodes pointed at by this one
}
```

### AOA event (a circle)

```jsonc
{
  "id": "1",                    // label of the event circle (or "E1")
  "position": [300.0, 100.0],
  "bounding_box": [265.0, 65.0, 70.0, 70.0]
}
```

### AOA arrow / activity

An arrow is an edge between two events. Store it **in the activity list**
(after the schema of the module), with the event ids as predecessors/successors:

```jsonc
{
  "id": "A",                    // activity name written on the arrow (optional)
  "duration": 5.0,              // duration written on the arrow (optional)
  "predecessors": ["S"],        // source event id
  "successors": ["1"],          // target event id
  "dummy": false
}
```

Alternatively list the arrow explicitly under `dependencies`:
```jsonc
{ "source": "S", "target": "1" }
```
(If `dependencies` is empty for an AOA file, they are derived from the
activities' event transitions.)

### Dependency (AON)

```jsonc
{ "source": "A", "target": "B" }  // directed: A precedes B
```

Coordinates are in **image pixel space** (x right, y down, origin top-left).
Geometry is optional — without it, node matching degrades to count-only
(precision/recall become `N/A`, which the report shows honestly).

---

## Draft workflow (quick path)

`export-draft` runs the real pipeline once and seeds a draft annotation per
image (ids, geometry, a dependency guess). It never decides truth — it just
saves you from typing coordinates and the obvious ids.

```sh
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" export-draft
```

Then for every file **except the gold** (`1.json`):

1. `python -m pert_analyzer.benchmark.annotate ... show <image>` and open the
   image next to it.
2. Correct OCR'd ids (they are often off by a letter, e.g. `INFERRED_..`).
   Repeated ids were auto-renamed `X_2`, `X_3`, … — verify they are really
   distinct nodes.
3. Add missing nodes/events (positions optional; without geometry the metric
   is count-only).
4. Fix the `dependencies`: the seeded list comes from the detector and is
   incomplete — compare against the arrows you see (for `1.png` the true list
   has 28 edges, the detector draft has 11).
5. Add durations where the diagram shows numbers.
6. Set the status and counts, then validate:

```sh
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" set-status 5.jpeg COMPLETE
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" set-expected 5.jpeg --activities 17 --dependencies 10
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" validate
```

`validate` exits 0 only when there are no errors, and still prints warnings
(e.g. a missing duration) that are worth fixing.

## CLI helpers

```sh
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" list
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" show 1.png
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" validate
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" set-status 1.png COMPLETE
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" set-duration 1.png A 5.0
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" set-expected 1.png --activities 22 --dependencies 28
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" add-activity 1.png A --duration 5.0
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" add-event 3.jpeg E1
python -m pert_analyzer.benchmark.annotate "tests/test_data/Imag PERT" add-dependency 1.png A B
```

The CLI only reads the dataset to locate images and writes these JSON files;
it never runs the CV pipeline.

---

## Arabic quick reference (الملخص)

- كل صورة لها ملف `ground_truth/<الاسم>.json` (نفس اسم الصورة بدون الامتداد).
- هذا المرجع **يدوي**: لا تُكتب من مخرجات المحلل، وإلا أصبح القياس دائريًا.
- `AON`: النشاطات عُقد `activities` بقائمة `successors/predecessors` حسب الاتجاه، و`dependencies` حافة موجهة `{source,target}`.
- `AOA`: الأحداث دوائر في `events`، والأسهم نشاطات في `activities` بـ`predecessors=[حدث المصدر]` و`successors=[حدث الهدف]` (أو `dependencies` صريحة).
- كل ما لا تستطيع قراءته بدقة: ضعه في `uncertain` (أو اجعل الحالة `UNCERTAIN`) ولا تخمّن أبدًا — البنمارك يستثني العناصر غير المؤكدة من الدقة.
- الإحداثيات بكسلات الصورة (x يمين، y أسفل، الأصل أعلى-يسار). بدون هندسة تُحسب الدقة counts-only.