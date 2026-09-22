# Phase 9 — Analysis-Completion Stack Overflow Fix (0xC00000FD)

**Status:** COMPLETE (root cause isolated, production fix landed in the review
widget, crash empirically erased on the real image with the real worker, and full
regression green).
**Scope gate:** one GUI widget class rewritten (`ImageContextView`) + one new
regression test. No CV / OCR / CPM / PERT algorithm code touched. No UI redesign,
no `try/except: pass` masking. Full repo regression: **1243 passed** in 02:11.

---

## 1. Problem / Symptom

After clicking **Analyze** in the real GUI, the application exited with
`exit code -1073741571` (`0xC00000FD` — *stack overflow*). The crash was:

- **async-only** — the same completion path executed with the backend on the main
  thread never crashed (2/2 runs); the live `QThread`-based worker flow crashed in
  the large majority of runs (baseline reproducer).
- nondeterministic in exact timing but structurally tied to the human-review
  completion path: consistent crashes when the completion handler rebuilt the
  Review page while it was live/visible; reliable survival when that page was
  rebuilt hidden or not navigated to.
- reported (via `faulthandler`) as occurring on the **main thread** with only a
  bare `<module>`/`app.exec()` Python frame — the overflowing frames were
  **pure native Qt** (`QLabel`/`QLayout`/`QWidget` geometry), so Python stack
  capture could not show the chain.

## 2. Root Cause

`pert_analyzer/gui/reviews/widgets.py`, class `ImageContextView`, coupled the
displayed image to layout geometry through a layout-managed `QLabel`:

```
resizeEvent(event) -> _repaint_scaled()
set_context(...)    -> _repaint_scaled()
_repaint_scaled()   -> self._image_label.setPixmap(self._pixmap.scaled(label.size(), KeepAspectRatio))
```

The crafted widget is placed inside the Activity/Dependency/Duration detail panels,
which live in a `QStackedWidget` inside the Review page. Once the Review page is
**visible** (i.e. after an *asynchronous* analysis completes, the main window has
been live for ~55–70 s and every widget has settled real geometry), every
`QLabel.setPixmap` call changes the label's size hint and emits
`QWidget::updateGeometry`, which synchronously re-activates the surrounding
`QLayout`. Because the label keeps a `MinimumHeight(180)` and the pixmap is
`KeepAspectRatio`-fitted, the layout delivers a fresh `QResizeEvent` back into
`ImageContextView`, whose `resizeEvent` handler calls `_repaint_scaled()` again,
re-setting a pixmap whose fitted size never equals the geometry just assigned.
Each iteration runs on the native Qt stack and never converges, so the stack grows
unboundedly until Windows terminates the process with `0xC00000FD`.

### Why async-only
- **Async (crash):** during the ~55–70 s analysis the window is live; when the
  queued completion arrives, the Review page is real, visible, and already laid
  out, so the `setPixmap → layout → resizeEvent → setPixmap` loop engages
  immediately with nonzero geometry.
- **Sync (no crash):** the main thread is blocked for the whole analysis, so when
  the handler runs, the window has no live geometry / the review page is not in a
  laid-out-visible state; the feedback loop never starts.
- **Hidden-page rebuild (no crash):** `ReviewPage.refresh()` on a page still inside
  the (hidden) `QStackedWidget` page does layout in a non-visible branch where the
  geometry feedback cannot cascade (empirically: `noreview` mode always survived).

## 3. Recursive Call Chain (native, inferred from behavioral bisection)

```
QApplication::exec
  └─ main-thread event delivery (queued "completed" QMetaCallEvent)
       MainWindow._on_analysis_completed            (main_window.py:360)
         ├─ _session.complete_analysis(workflow)
         └─ _refresh_pages()  →  ReviewPage.refresh  →  _rebuild  →  _populate
            MainWindow._on_nav(REVIEW)               (main_window.py:285)
              ├─ self._stack.setCurrentIndex(REVIEW)   ← page becomes VISIBLE
              └─ page.refresh()  →  _present()  →  _on_item_picked()
                   ActivityDetailPanel.show_item        (detail_panels.py:269)
                     └─ _render_context()  →  ImageContextView.set_context
                          (widgets.py, OLD)  →  _repaint_scaled()
                               ├─ label.setPixmap(scaled(self._pixmap))
                               │     └─ QLabel::setPixmap → QWidget::updateGeometry
                               │          └─ QBoxLayout::activate (synchronous)
                               │               └─ sendEvent(QEvent::Resize)
                               └─ ImageContextView::resizeEvent
                                     └─ _repaint_scaled() → label.setPixmap(…)  →  …∞repeat
   → native stack exhaustion → 0xC00000FD (no Python frames on the overflowed stack)
```

The key insight is that `resizeEvent` *feeds* the pixmap update and the pixmap
update *feeds* the resize — a mutual-synchronous feedback loop with no convergence
guarantee, running on the C++ stack.

## 4. Responsible File / Function

| Item | Location |
|---|---|
| Widget class with the loop | `pert_analyzer/gui/reviews/widgets.py` — `ImageContextView` |
| OLD overflow path | `ImageContextView.__init__` (layout-managed `QLabel`), `_set_placeholder`, `clear`, `set_context`, `_repaint_scaled`, `resizeEvent` |
| Panel that activates it on visible page | `pert_analyzer/gui/reviews/detail_panels.py:269` (`ActivityDetailPanel.show_item` → `_render_context`) |
| Asynchronous completion driver | `pert_analyzer/gui/worker.py` (`AnalysisWorker`), `pert_analyzer/gui/main_window.py:360` (`_on_analysis_completed`) |

## 5. Production Fix

Rewrote `ImageContextView` as a **self-painting widget** (`pert_analyzer/gui/reviews/widgets.py`):

- Removed the layout-managed `QLabel` and hence `setPixmap`/`setText` on a layout child.
- The overlay is rendered directly in a new `paintEvent`: it fits the stored
  `QPixmap` to the view rect (`KeepAspectRatio` + `SmoothTransformation`), centers
  it, and draws the dashed-border placeholder text when no context is set.
- `resizeEvent` and `_repaint_scaled` are gone; resizing only invokes `update()`
  via normal Qt mechanics; the drawn content never participates in layout size
  negotiation, so the geometry↔content feedback loop cannot start.

Public surface is unchanged — `set_context(image_path, highlights, image_size)`,
`clear()`, `highlight_count`, `current_highlights`, `_pixmap` — so the three detail
panels and all existing tests keep working. Visual result is equivalent (image
scales to fit in the card, placeholder styling preserved).

## 6. Diagnostic Evidence

Behavioral bisection (each ~55–70 s real-OCR run, `0xC00000FD` observed as
`EXITCODE=-1073741571`; stdout markers via `faulthandler` + tick/state logging):

| Experiment | Variable changed | Result |
|---|---|---|
| backend-only (`ReviewWorkflow.analyze`, no GUI) | — | exit 0 (backend immune) |
| `sync_full` (analyze on main thread, full handler), 2 runs | threading | exit 0, exit 0 |
| `async_full` (analysis QThread, full handler) — baseline | — | crash (reproducible) |
| `async_nullw` (worker emits `None`) | payload | no crash |
| `async_payload_ignore` (real payload, handler ignores) | payload | no crash |
| `async_full_keepalive` (worker kept alive after cleanup) | worker GC | crash |
| `async_noemit` (worker never emits completed) | signal | no crash |
| `experiment_store` (result attr, no-arg signal) | QObject<->object marshalling | crash (payload theory dead) |
| `watchdog` (2 s dumps + lifetime marker) | timing | `HANDLER_DONE t=54.872s` → crash ≤0.5 s later, no `FINAL` marker; main thread idle-at-`<module>` in all dumps → native recursion |
| `handler_dump` (0.15 s dumps in handler) | stack sampling | bounded Python frames inside handler (`PIL.Image.open`, `_render_overlay`), `HANDLER DONE in 0.12s` → crash uses NO Python frames |
| re-entry harness (guards around callback re-entry) | Python re-entrancy | max per-key depth 1, max total depth 8, **0 runaway** |
| `experiment_updates` (`setUpdatesEnabled(False)`) | paint events | still crash → layout/geometry loop, not painting |
| `experiment_nofinished` (no `finished`→`_cleanup_worker` slot) | cleanup timing | still crash |
| `experiment_cuts` MODE=noreview (skip only `_on_nav(REVIEW)`) | review-page visibility | exit 0 (page hidden refresh safe) |
| `experiment_cuts` MODE=full (faithful handler) | — | crash (control) |
| `experiment_skip` (skip `panel.show_item` only) | detail-panel content | exit 0 → crash is inside `show_item` |
| `experiment_noscal` (`setPixmap` once, `resizeEvent` no-op) | pixmap↔resize coupling | exit 0 → coupling is the trigger |
| `experiment_guard` (size-match guard on `_repaint_scaled`), 3 runs | convergence guard | crash x2, OK x1 → non-converging churn; simple guard insufficient |
| `experiment_paintview` (paintEvent-only widget), 3 runs | widget design | exit 0 x3 → final fix validated |

**Timeline (watchdog run):** `START t=0.441s` → `WORKER_ANALYZE_RETURNED t=54.750s`
→ `HANDLER_START t=54.751s` → `HANDLER_DONE t=54.872s` → crash within ~0.2 s.
Everything after the handler is main-thread Qt work — fully consistent with the
post-completion review-page repaint/layout burst.

## 7. Regression Test Added

`tests/gui/test_review_center.py`
`test_image_context_view_survives_resize_with_live_geometry` — builds a visible
`ImageContextView` inside a live layout, calls `set_context` with a real highlight
overlay, then repeatedly resizes the host (the geometry-feedback scenario from the
bug) and processes events. Asserts the view still shows the overlay (`_pixmap` set,
`highlight_count == 1`) and structurally pins the fix (`_image_label` and
`_repaint_scaled` are gone).

Targeted GUI slice (review center + analysis page): **41 passed in 5.20 s**.

## 8. Full Regression

`python -m pytest tests/ -q` (headless, `QT_QPA_PLATFORM=offscreen`):

```
1243 passed, 2 warnings in 131.69s (0:02:11)
```

(1242 pre-existing + 1 new regression test; only deprecation warnings from the
unrelated PDF export font API remain.)

## 9. Real-Image Smoke Test + Exit Code

Real reference AON diagram (`tests/test_data/reference_diagrams/reference_aon.png`),
real `ReviewWorkflow.analyze` backend in the real `AnalysisWorker` QThread, real
`MainWindow._on_analysis_completed` path, `python -X faulthandler`:

```
SMOKE RUN 1 OK    (state=ANALYZING … REVIEW_REQUIRED … SMOKE COMPLETED)
SMOKE RUN 2 OK
Overall fail: 0
```

Additional fix-validation run with REVIEW→ANALYZE→REVIEW page round-trip:
3/3 exit 0 (also exercises the Analysis page image view re-show after analysis).

**Exit code after the fix: `0`** on every real-image completion run, where the
pre-fix baseline exited `-1073741571`.

## 10. Confirmation the Overflow Is Gone

- Pre-fix baseline (`async_full`, faithful handler): `-1073741571` (`0xC00000FD`)
  in the majority of runs, reliably reproducible across the QThread/guard/store/
  keepalive/no-finished variants.
- Post-fix (production code, `validate_fix.py` and `smoke_fix.py`):
  **5 completed real-image runs, all `exit 0`**, including navigation back and
  forth between pages after completion.
- The mechanism is eliminated structurally (no layout-child pixmap, no
  `resizeEvent`-fed repaint), not masked: `ImageContextView` can no longer
  participate in the layout geometry feedback that consumed the native stack.