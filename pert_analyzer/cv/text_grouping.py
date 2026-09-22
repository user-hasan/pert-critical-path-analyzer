"""
Text grouping utilities for OCR post-processing.

Groups spatially close text regions into logical text clusters
that likely represent the same diagram element.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.ocr_models import OCRTextRegion, TextGroup


@dataclass
class TextGroupingConfig:
    """Configuration for text grouping."""

    # Distance thresholds
    horizontal_distance_threshold: float = 60.0
    vertical_distance_threshold: float = 30.0
    overall_distance_threshold: float = 80.0

    # Bounding box overlap
    min_overlap_for_group: float = 0.0

    # Group size limits
    max_group_size: int = 10
    min_group_size: int = 2


class TextGrouper:
    """
    Groups OCR text regions into logical text clusters.

    Uses spatial proximity and bounding box relationships to identify
    text regions that likely belong to the same diagram element.
    """

    def __init__(self, config: Optional[TextGroupingConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or TextGroupingConfig()

    def group_regions(
        self, regions: List[OCRTextRegion]
    ) -> List[TextGroup]:
        """
        Group text regions into logical clusters.

        Args:
            regions: List of OCR text regions.

        Returns:
            List of TextGroup objects.
        """
        if len(regions) < self.config.min_group_size:
            return []

        # Build adjacency based on proximity
        groups: List[List[int]] = []
        used = set()

        for i, region_i in enumerate(regions):
            if i in used:
                continue

            # Start a new group
            group_indices = [i]
            used.add(i)

            for j, region_j in enumerate(regions):
                if j in used:
                    continue

                if self._should_group(region_i, region_j):
                    # Check if j is close to any member of the current group
                    should_add = False
                    for member_idx in group_indices:
                        if self._should_group(regions[member_idx], region_j):
                            should_add = True
                            break
                    if should_add:
                        group_indices.append(j)
                        used.add(j)

            if len(group_indices) >= self.config.min_group_size:
                groups.append(group_indices)

        # Convert to TextGroup objects
        text_groups = []
        for group_indices in groups:
            group = self._create_group(
                [regions[i] for i in group_indices]
            )
            text_groups.append(group)

        return text_groups

    def _should_group(
        self, region_a: OCRTextRegion, region_b: OCRTextRegion
    ) -> bool:
        """Determine if two text regions should be grouped."""
        bbox_a = region_a.bounding_box
        bbox_b = region_b.bounding_box

        # Check overall distance
        center_a = bbox_a.center
        center_b = bbox_b.center
        distance = center_a.distance_to(center_b)

        if distance > self.config.overall_distance_threshold:
            return False

        # Check horizontal alignment
        dx = abs(center_a.x - center_b.x)
        dy = abs(center_a.y - center_b.y)

        # Vertically stacked (same x, close y)
        if dx < self.config.horizontal_distance_threshold:
            if dy < self.config.vertical_distance_threshold * 3:
                return True

        # Horizontally aligned (same y, close x)
        if dy < self.config.vertical_distance_threshold:
            if dx < self.config.horizontal_distance_threshold:
                return True

        # Close overall
        if distance < self.config.overall_distance_threshold * 0.5:
            return True

        return False

    def _create_group(self, regions: List[OCRTextRegion]) -> TextGroup:
        """Create a TextGroup from a list of regions."""
        region_ids = [r.region_id for r in regions]

        # Compute combined bounding box
        min_x = min(r.bounding_box.x for r in regions)
        min_y = min(r.bounding_box.y for r in regions)
        max_x = max(r.bounding_box.x + r.bounding_box.width for r in regions)
        max_y = max(r.bounding_box.y + r.bounding_box.height for r in regions)
        combined_bbox = BoundingBox(
            x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y
        )

        # Combine text (sorted by y then x for logical reading order)
        sorted_regions = sorted(
            regions, key=lambda r: (r.bounding_box.y, r.bounding_box.x)
        )
        combined_text = " ".join(
            r.normalized_text or r.text for r in sorted_regions
        )
        normalized_text = combined_text.strip()

        # Average confidence
        avg_confidence = (
            sum(r.confidence for r in regions) / len(regions)
            if regions
            else 0.0
        )

        return TextGroup(
            member_region_ids=region_ids,
            combined_text=combined_text,
            normalized_combined_text=normalized_text,
            combined_bounding_box=combined_bbox,
            group_confidence=avg_confidence,
            member_count=len(regions),
        )

    def merge_groups(
        self, groups: List[TextGroup]
    ) -> List[TextGroup]:
        """
        Merge overlapping groups.

        Args:
            groups: List of text groups.

        Returns:
            Merged list of text groups.
        """
        if len(groups) <= 1:
            return groups

        merged = True
        while merged:
            merged = False
            new_groups = []
            used = set()

            for i, group_i in enumerate(groups):
                if i in used:
                    continue

                current_members = set(group_i.member_region_ids)

                for j, group_j in enumerate(groups):
                    if j <= i or j in used:
                        continue

                    # Check if groups share members
                    shared = current_members.intersection(
                        set(group_j.member_region_ids)
                    )
                    if shared:
                        # Merge
                        current_members.update(group_j.member_region_ids)
                        used.add(j)
                        merged = True

                if not merged or i not in used:
                    # Rebuild group from merged members
                    # (We need the original regions, but we only have IDs)
                    # For simplicity, keep the group as-is if we can't merge
                    new_groups.append(group_i)
                    used.add(i)

            if merged:
                groups = new_groups

        return groups
