"""
Deterministic synthetic image generation for end-to-end testing.

Creates test images with known ground truth for validating
the complete pipeline without relying on real diagram images.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False


def create_simple_chain_image(
    width: int = 800,
    height: int = 200,
    box_width: int = 100,
    box_height: int = 60,
    activities: Optional[List[Dict[str, Any]]] = None,
    bg_color: int = 255,
    border_color: int = 0,
    text_color: int = 0,
    arrow_color: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a synthetic AON image with activities in a chain.

    Creates: [A:5] -> [B:3] -> [C:4] (or custom activities)

    Returns:
        Tuple of (image_array, ground_truth_dict).
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for image generation")

    if activities is None:
        activities = [
            {"id": "A", "duration": 5, "label": "A"},
            {"id": "B", "duration": 3, "label": "B"},
            {"id": "C", "duration": 4, "label": "C"},
        ]

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    n = len(activities)
    spacing = width // (n + 1)
    y_center = height // 2
    boxes = []

    for i, act in enumerate(activities):
        x = spacing * (i + 1) - box_width // 2
        y = y_center - box_height // 2
        boxes.append((x, y, box_width, box_height))

        cv2.rectangle(img, (x, y), (x + box_width, y + box_height), border_color, 2)

        id_text = act["id"]
        cv2.putText(img, id_text, (x + 10, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        dur_text = str(act["duration"])
        cv2.putText(img, dur_text, (x + 10, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        y1 = y_center
        x2 = boxes[i + 1][0]
        y2 = y_center
        cv2.arrowedLine(img, (x1, y1), (x2, y2), arrow_color, 2, tipLength=0.15)

    ground_truth = {
        "diagram_type": "AON",
        "activities": [
            {"id": act["id"], "duration": act["duration"], "label": act.get("label", act["id"])}
            for act in activities
        ],
        "dependencies": [
            {"source": activities[i]["id"], "target": activities[i + 1]["id"]}
            for i in range(len(activities) - 1)
        ],
        "expected_project_duration": sum(a["duration"] for a in activities),
        "expected_critical_path": [a["id"] for a in activities],
    }

    return img, ground_truth


def create_branching_image(
    width: int = 900,
    height: int = 400,
    bg_color: int = 255,
    border_color: int = 0,
    text_color: int = 0,
    arrow_color: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a synthetic AON image with branching paths.

    Creates:
        A(2) -> B(4) -> D(3)
        A(2) -> C(1) -> D(3)

    Returns:
        Tuple of (image_array, ground_truth_dict).
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for image generation")

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    box_w, box_h = 100, 60

    positions = {
        "A": (50, 150),
        "B": (300, 50),
        "C": (300, 250),
        "D": (600, 150),
    }
    activities = [
        {"id": "A", "duration": 2},
        {"id": "B", "duration": 4},
        {"id": "C", "duration": 1},
        {"id": "D", "duration": 3},
    ]

    boxes = {}
    for act in activities:
        act_id = act["id"]
        x, y = positions[act_id]
        boxes[act_id] = (x, y, box_w, box_h)
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), border_color, 2)
        cv2.putText(img, act_id, (x + 10, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(img, str(act["duration"]), (x + 10, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    arrows = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    for src, tgt in arrows:
        sx = boxes[src][0] + boxes[src][2]
        sy = boxes[src][1] + boxes[src][3] // 2
        tx = boxes[tgt][0]
        ty = boxes[tgt][1] + boxes[tgt][3] // 2
        cv2.arrowedLine(img, (sx, sy), (tx, ty), arrow_color, 2, tipLength=0.12)

    ground_truth = {
        "diagram_type": "AON",
        "activities": activities,
        "dependencies": [{"source": s, "target": t} for s, t in arrows],
        "expected_project_duration": 9,
        "expected_critical_path": ["A", "B", "D"],
    }

    return img, ground_truth


def create_blank_image(
    width: int = 400,
    height: int = 300,
    color: int = 255,
) -> np.ndarray:
    """Create a blank white image for negative testing."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def create_noisy_image(
    width: int = 400,
    height: int = 300,
    noise_level: int = 50,
) -> np.ndarray:
    """Create a random noise image for negative testing."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def save_synthetic_image(
    image: np.ndarray,
    filepath: str,
) -> str:
    """
    Save a synthetic image to disk.

    Returns:
        The saved file path.
    """
    import os

    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for image saving")

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    cv2.imwrite(filepath, image)
    return filepath
