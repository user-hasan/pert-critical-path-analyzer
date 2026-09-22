"""
Deterministic dataset scanning for the generalization benchmark.

The benchmark must never reorder or rename the user's dataset.  This
module only *reads* the tree, filters to supported raster image
formats, and returns a deterministically sorted list of relative paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

# Supported raster formats for the benchmark corpus.  Everything else
# (JSON/TXT/CSV/PDF/reports/non-image binaries) is ignored silently.
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


class DatasetScanner:
    """Recursive, deterministic image-file scanner for a dataset root."""

    def __init__(
        self,
        dataset_root: str | Path,
        extensions: Iterable[str] = SUPPORTED_IMAGE_EXTENSIONS,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self._suffixes = {e.lower() for e in extensions}

    def discover(self) -> List[Path]:
        """Return all supported image files, sorted deterministically.

        Ordering is a case-folded, platform-normalized (posix) sort of
        the relative path, so results are identical on any OS.
        """
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root not found or not a directory: {self.dataset_root}"
            )
        images: List[Path] = []
        for candidate in self.dataset_root.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in self._suffixes:
                continue
            images.append(candidate)
        images.sort(key=lambda p: p.relative_to(self.dataset_root).as_posix().lower())
        return images


def discover_images(
    dataset_root: str | Path,
    extensions: Iterable[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> List[Path]:
    """Convenience wrapper around :class:`DatasetScanner`."""
    return DatasetScanner(dataset_root, extensions=extensions).discover()


def relative_image_path(
    dataset_root: str | Path, image_path: str | Path
) -> str:
    """Return the dataset-relative posix path for an image.

    Raises ValueError when ``image_path`` is not inside ``dataset_root``.
    """
    root = Path(dataset_root).resolve()
    image = Path(image_path).resolve()
    try:
        return image.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(
            f"Image {image} is not inside dataset root {root}"
        ) from None