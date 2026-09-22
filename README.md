# PERT & Critical Path Analyzer

[![CI](https://github.com/user-hasan/pert-critical-path-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/user-hasan/pert-critical-path-analyzer/actions/workflows/ci.yml)

> Desktop application that analyzes project network diagrams (AON / AOA) from images using Computer Vision + OCR, then computes the Critical Path (CPM) and PERT estimates.
> Supervisor: Dr. Adel Al-Afeery

## الفكرة باختصار

صورة مخطط مشروع ← معالجة صورة ← كشف أشكال وأسهم ← OCR ← ربط مكاني ← إعادة بناء دلالية ← `GraphModel` ← تحليل CPM/PERT ← مراجعة بشرية ← تقرير PDF/Excel/JSON/CSV.

## Architecture

```
GUI (PySide6): Analysis / Understanding / Review / Validation / Results / NetworkBuilder
        │ QThread Workers + Qt Signals/Slots
Pipeline: EndToEndAnalyzer (12 stages: load → preprocess → shapes → classify → arrows → OCR → associate → reconstruct → graph → CPM → review → export)
        │
Domain: core (models/interfaces) + graph (builder/adapter) + analysis (cpm_engine/pert_engine) + validation
        │
CV: preprocessing → shape_detection → classification → arrow_detection → ocr_engine → spatial_association → reconstruction → reconciliation
        │
Infra: config + persistence + reporting/export + benchmark
```

Key principle: detectors produce **evidence with confidence**, the `ReconstructionEngine` makes the semantic decision. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e ".[all]"

# Tesseract OCR (optional, for real OCR instead of mock):
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt install tesseract-ocr tesseract-ocr-ara
```

## Usage

```bash
# CLI pipeline
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png"
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --output result.json
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --debug

# GUI
python -m pert_analyzer.gui

# Benchmark
python -m pert_analyzer.benchmark
```

## Tests

```bash
pytest
```

## Project structure

```
pert_analyzer/
  core/          # dataclasses (GraphModel, Activity, AnalysisResult...) + ABC interfaces
  cv/            # preprocessing, shape/arrow detection, OCR, association, reconstruction
  graph/         # GraphBuilder + NetworkXAdapter
  analysis/      # CPMEngine + PertEngine
  pipeline/      # EndToEndAnalyzer (12 stages), PipelineResult, human_review, review_api, debug_export
  gui/           # PySide6 app: pages, workers, session, themes, results, reviews, builder
  reporting/     # ReportBuilder (Qt-free) + export (pdf/excel/json/csv)
  persistence/   # JSON + CSV import/export of graphs
  benchmark/     # dataset runner + metrics + markdown reports
  config/        # AppConfig + ConfigManager (see config.json)
  validation/    # structural/data/confidence checks
scripts/
  debug/         # legacy debug/integration scripts (moved from repo root)
tests/           # unit + integration + gui + cv tests
docs/            # phase reports (PHASE6..9, acceptance, generalization benchmark)
artifacts/       # reference validation artifacts
```

## Configuration

See [config.json](config.json): OCR engine/languages, CV thresholds, analysis engine, GUI theme/size, persistence paths, logging.

## Reports

- [ARCHITECTURE.md](ARCHITECTURE.md) — full architecture document
- [PIPELINE_REPORT.md](PIPELINE_REPORT.md) — latest end-to-end run results
- [docs/](docs) — phase and acceptance reports

## License

MIT — see [LICENSE](LICENSE).
