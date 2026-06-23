# Cell Plotter — static visualizer for the eyeball generation probe

An interactive, **local, no-model** visualizer for the 21 pre-generated cells in
`reports/_eyeball_probe/`. It shows that the token-only model draws **valid, coherent,
conditioning-responsive** city geometry — the methodology-validation question
(`docs/PROJECT_FOCUS.md`), **not** a realism or architecture claim.

## How to open it

The site is pure static files with the data inlined as a `<script>`, so just open it —
**no server needed**:

```
open viz/index.html        # macOS
```

(Or double-click `viz/index.html` in Finder.) A modern browser is required; fonts load
from Google Fonts (a system fallback is used offline).

If your browser blocks local `file://` for any reason, serve the folder instead:

```
python3 -m http.server -d viz 8000   # then open http://localhost:8000
```

## What you can do

- **Inspect** — pick one of the three density contexts (dense urban / medium-mixed /
  sparse suburban) and one of its 7 cells. The geometry plots onto a coordinate grid;
  scroll to zoom, drag to pan. The right rail shows the **conditioning** the model was
  given, the **character-stats vector** fed in (flagged honestly as an echo), and the
  **output** (token count, self-termination, decode rate, per-class counts). Click a
  legend chip to toggle a feature class.
- **Compare** — the three contexts side by side at one shared scale, so the density
  response is visually obvious, plus the per-context median table.

## Feature classes (why color matters)

Features are classified by **construction identity** (the grammar's building-class
tokens via `cfm.eval.geometry`), not by shape:

| class | meaning | drawn as |
|---|---|---|
| `building · sealed` | building block, ring closes exactly | filled coral polygon |
| `building · near-closed` | building block, ring nearly closes | open dashed coral outline |
| `road` | non-building line | cyan line |
| `road node` | non-building point | amber dot |

The probe's raw GeoJSON labels near-closed building footprints as `road` LineStrings
(they fail an over-strict exact-closure check — a **deferred** eval-side defect, see
`PROJECT_FOCUS.md`). This viz recovers the true class so those footprints are not
misread as roads — and surfaces the closure gap on hover.

## Regenerating the data bundle

`viz/data.js` is generated from the immutable probe artifacts. `reports/` is never
written. To rebuild after a new probe:

```
uv run python viz/build_data.py
```

A regression test pins the faithful class counts:

```
uv run pytest tests/test_viz_build_data.py -q
```
