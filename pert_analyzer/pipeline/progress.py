"""
Structured pipeline progress events shared by the analyzer and the GUI.

The analyzer already reports ``(stage, progress)`` through its classic
``progress_callback`` channel for terminal/CLI consumers. The GUI needs
per-stage state, a human message, and detected-object metrics, so the
analyzer also accepts an optional ``stage_callback`` that receives
:class:`StageProgress` events. The two channels are independent:
adding the structured channel never changes the values a plain
``progress_callback`` receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Canonical pipeline stage ids, emitted at real pipeline events in the
# analyzer's actual execution order. Never invent a percentage here:
# every event corresponds to a stage that really runs.
PIPELINE_STAGES: tuple[str, ...] = (
    "Loading image",
    "Preprocessing",
    "Detecting shapes",
    "Classifying diagram",
    "Detecting arrows",
    "Extracting text (OCR)",
    "Associating text",
    "Reconstructing diagram",
    "Building graph",
    "Validating and analyzing",
    "Complete",
    "Review preparation",
)

# Human-facing labels for the stage ids above.
STAGE_LABELS: Dict[str, str] = {
    "Loading image": "Loading Image",
    "Preprocessing": "Preprocessing",
    "Detecting shapes": "Shape Detection",
    "Classifying diagram": "Diagram Classification",
    "Detecting arrows": "Arrow Detection",
    "Extracting text (OCR)": "OCR",
    "Associating text": "Spatial Association",
    "Reconstructing diagram": "Reconstruction",
    "Building graph": "Semantic Resolution",
    "Validating and analyzing": "Graph Validation",
    "Complete": "Finalization",
    "Review preparation": "Review Preparation",
}

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"

STAGE_STATES: tuple[str, ...] = (
    PENDING,
    RUNNING,
    COMPLETED,
    REVIEW_REQUIRED,
    FAILED,
    SKIPPED,
)


@dataclass
class StageProgress:
    """One progress event for a single pipeline stage.

    ``progress`` is the overall pipeline progress (0.0 .. 1.0) reported
    by the analyzer at the exact moment the event fires. ``metrics``
    carries only real, measured values (shape count, arrow count, OCR
    region count, reconstructed activities, review-item count).
    """

    stage_id: str
    state: str
    progress: float
    name: str = ""
    message: str = ""
    metrics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = STAGE_LABELS.get(self.stage_id, self.stage_id)