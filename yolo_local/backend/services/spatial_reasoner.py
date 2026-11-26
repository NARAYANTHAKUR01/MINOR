"""
Spatial Reasoning & Location Response Generator.

Calculates normalized 3x3 frame coordinates for detected bounding boxes:
- Grid layout:
    top-left    | top    | top-right
    left        | center | right
    bottom-left | bottom | bottom-right

Matches requested regions (broad or specific) and formats natural-language spoken feedback.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Sequence
from backend.schemas.detection import BoundingBox, DetectionItem


# ---------------------------------------------------------------------------
# Spatial Direction & Environmental Perception Constants
# ---------------------------------------------------------------------------
VALID_REGIONS = {
    "left", "right", "top", "bottom", "center",
    "top-left", "top-right", "bottom-left", "bottom-right",
    "middle", "middle-left", "middle-right"
}

BROAD_DIRECTION_MAP: dict[str, set[str]] = {
    "left": {"left", "top-left", "bottom-left"},
    "right": {"right", "top-right", "bottom-right"},
    "top": {"top", "top-left", "top-right"},
    "bottom": {"bottom", "bottom-left", "bottom-right"},
    "center": {"center", "middle"},
    "middle": {"center", "middle"},
    "top-left": {"top-left"},
    "top-right": {"top-right"},
    "bottom-left": {"bottom-left"},
    "bottom-right": {"bottom-right"},
}

ENVIRONMENT_DIRECTION_MAP: dict[str, str] = {
    "center": "directly in front of you",
    "middle": "directly in front of you",
    "left": "on your left",
    "right": "on your right",
    "top": "up ahead in front of you",
    "bottom": "down in front of you",
    "top-left": "up ahead on your left",
    "top-right": "up ahead on your right",
    "bottom-left": "down on your lower left",
    "bottom-right": "down on your lower right",
}


class SpatialReasoner:
    """
    Performs spatial analysis on detected bounding boxes relative to frame dimensions.
    """

    @staticmethod
    def calculate_object_location(
        bbox: BoundingBox | dict[str, float],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        """
        Calculates normalized center coordinates and resolves the 3x3 grid region.

        Args:
            bbox: BoundingBox object or dict with x1, y1, x2, y2.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            dict containing:
                - center_x, center_y (pixels)
                - norm_x, norm_y (0.0 to 1.0)
                - region (e.g. 'left', 'top-right', 'center')
                - friendly_phrase (e.g. 'on the left side of the frame')
        """
        if isinstance(bbox, dict):
            x1 = bbox.get("x1", 0.0)
            y1 = bbox.get("y1", 0.0)
            x2 = bbox.get("x2", 0.0)
            y2 = bbox.get("y2", 0.0)
        else:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        w = max(frame_width, 1)
        h = max(frame_height, 1)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        nx = max(0.0, min(1.0, cx / w))
        ny = max(0.0, min(1.0, cy / h))

        # 3x3 grid partitioning
        # Horizontal: Left (< 0.35), Center (0.35 to 0.65), Right (> 0.65)
        # Vertical:   Top (< 0.35), Middle (0.35 to 0.65), Bottom (> 0.65)
        if nx < 0.35:
            h_pos = "left"
        elif nx > 0.65:
            h_pos = "right"
        else:
            h_pos = "center"

        if ny < 0.35:
            v_pos = "top"
        elif ny > 0.65:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # Combine into cohesive environmental perception phrase
        if v_pos == "middle" and h_pos == "center":
            region = "center"
            friendly = "directly in front of you"
        elif v_pos == "middle":
            region = h_pos  # 'left' or 'right'
            friendly = f"on your {h_pos}"
        elif h_pos == "center":
            region = v_pos  # 'top' or 'bottom'
            friendly = "up ahead in front of you" if v_pos == "top" else "down in front of you"
        else:
            region = f"{v_pos}-{h_pos}"  # e.g. 'top-left', 'bottom-right'
            if v_pos == "top":
                friendly = f"up ahead on your {h_pos}"
            else:
                friendly = f"down on your lower {h_pos}"

        return {
            "center_x": round(cx, 1),
            "center_y": round(cy, 1),
            "norm_x": round(nx, 3),
            "norm_y": round(ny, 3),
            "col": h_pos,
            "row": v_pos,
            "region": region,
            "friendly_phrase": friendly,
        }

    @classmethod
    def to_environment_direction(cls, region: str | None) -> str:
        """
        Converts region identifier to natural human environmental perspective.
        e.g. 'left' -> 'on your left', 'center' -> 'directly in front of you'.
        """
        if not region:
            return "in front of you"
        reg = region.strip().lower()
        return ENVIRONMENT_DIRECTION_MAP.get(reg, f"on your {reg}")

    @staticmethod
    def matches_requested_region(detected_region: str, requested_region: str | None) -> bool:
        """
        Checks if detected 3x3 region satisfies the user's requested region.
        Supports both broad ('left' matches 'top-left') and specific regions.
        """
        if not requested_region:
            return True

        req = requested_region.strip().lower()
        det = detected_region.strip().lower()

        if req == det:
            return True

        allowed = BROAD_DIRECTION_MAP.get(req)
        if allowed and det in allowed:
            return True

        return False

    @classmethod
    def generate_location_response(
        cls,
        target_name: str | None = None,
        detected_item: DetectionItem | dict[str, Any] | None = None,
        frame_shape: tuple[int, int] | None = None,
        requested_region: str | None = None,
        timed_out: bool = False,
        timeout_seconds: float = 5.0,
        mismatched_region: str | None = None,
        detected_region: str | None = None,
        region_matched: bool = True,
        found: bool | None = None,
        target_object: str | None = None,
    ) -> str:
        """
        Generates natural English responses for all detection scenarios in reference
        to human environmental perception (e.g. 'on your left', 'directly in front of you').
        """
        name = target_name or target_object or "object"
        target_display = name.strip().lower()

        # If found is explicitly False or timed_out is True and not found
        if found is False or (timed_out and not detected_region and not detected_item):
            return "Sorry, I can't detect this type of object around you."

        # If detected_region was provided directly
        if detected_region is not None:
            det_region = detected_region
            phrase = cls.to_environment_direction(det_region)
        elif detected_item is not None:
            if isinstance(detected_item, dict):
                bbox = detected_item.get("bbox", {})
            else:
                bbox = detected_item.bbox

            w = frame_shape[1] if frame_shape and len(frame_shape) >= 2 else 640
            h = frame_shape[0] if frame_shape and len(frame_shape) >= 2 else 480

            spatial_info = cls.calculate_object_location(bbox, w, h)
            det_region = spatial_info["region"]
            phrase = spatial_info["friendly_phrase"]
        else:
            return "Sorry, I can't detect this type of object around you."

        # Case: Requested region mismatch
        is_mismatch = (
            (region_matched is False)
            or (requested_region and not cls.matches_requested_region(det_region, requested_region))
            or (mismatched_region is not None)
        )
        if is_mismatch and requested_region:
            detected_dir = cls.to_environment_direction(mismatched_region or det_region)
            requested_dir = cls.to_environment_direction(requested_region)
            return (
                f"I found your {target_display} {detected_dir}, "
                f"but I couldn't find one {requested_dir}."
            )

        # Case: Successful detection (region matched or no region requested)
        return f"I found your {target_display} {phrase}."

    # ---------------------------------------------------------------------------
    # Anchor-Based Relative Spatial Reasoning (Target vs Reference Object)
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_centroid(bbox: BoundingBox | dict[str, float]) -> tuple[float, float]:
        """
        Calculate the 2D centroid (center_x, center_y) of a bounding box.
        Formula:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
        """
        return calculate_centroid(bbox)

    @staticmethod
    def determine_relative_direction(
        target_center: tuple[float, float],
        anchor_center: tuple[float, float],
        tolerance_x: float = 25.0,
        tolerance_y: float = 25.0,
    ) -> str:
        """
        Compare target centroid with anchor centroid using configurable tolerances:
          - target center_x < anchor center_x - tolerance_x: target is left of anchor
          - target center_x > anchor center_x + tolerance_x: target is right of anchor
          - target center_y < anchor center_y - tolerance_y: target is above anchor
          - target center_y > anchor center_y + tolerance_y: target is below anchor
        """
        return determine_relative_direction(
            target_center=target_center,
            anchor_center=anchor_center,
            tolerance_x=tolerance_x,
            tolerance_y=tolerance_y,
        )

    @staticmethod
    def select_anchor_object(
        target_item: DetectionItem,
        all_detections: Sequence[DetectionItem],
        target_class: str,
        config: "AnchorSelectionConfig | None" = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> DetectionItem | None:
        """
        Select a suitable detected object as the reference/anchor object.
        Prefers clear, nearby objects that are not the target itself.
        """
        return select_anchor_object(
            target_item=target_item,
            all_detections=all_detections,
            target_class=target_class,
            config=config,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    @classmethod
    def analyze_relative_spatial_relations(
        cls,
        target_class: str,
        detections: Sequence[DetectionItem],
        frame_shape: tuple[int, int] | None = None,
        tolerance_x: float = 25.0,
        tolerance_y: float = 25.0,
        anchor_config: "AnchorSelectionConfig | None" = None,
    ) -> "RelativeSpatialResult":
        """
        Performs end-to-end anchor-based spatial reasoning:
        1. Selects detections matching the target class (handles multiple targets clearly).
        2. Selects the most suitable reference/anchor object.
        3. Computes centroids and determines relative directions (left/right/above/below).
        4. Generates natural language spatial description.
        """
        return analyze_relative_spatial_relations(
            target_class=target_class,
            detections=detections,
            frame_shape=frame_shape,
            tolerance_x=tolerance_x,
            tolerance_y=tolerance_y,
            anchor_config=anchor_config,
        )

    @classmethod
    def locate_target_relative_to_anchor(
        cls,
        target_class: str,
        frame: Any = None,
        detections: Sequence[DetectionItem] | None = None,
        detector: Any = None,
        conf_threshold: float = 0.35,
        tolerance_x: float = 25.0,
        tolerance_y: float = 25.0,
        anchor_config: "AnchorSelectionConfig | None" = None,
    ) -> "RelativeSpatialResult":
        """
        High-level pipeline method:
        1. Accepts target class and either pre-computed detections or a camera/image frame.
        2. If frame is provided and detections is None, runs the detector on the frame.
        3. Runs anchor-based spatial reasoning and returns RelativeSpatialResult.
        """
        if detections is None:
            if frame is None:
                raise ValueError("Either 'detections' or 'frame' must be provided.")
            if detector is None:
                from backend.services.detector import get_default_detector
                detector = get_default_detector()

            # Run detector on frame to get all candidate objects (target + anchors)
            raw_detections, _, _ = detector.detect(
                frame,
                conf_threshold=conf_threshold,
                annotate=False,
            )
            active_detections = raw_detections
        else:
            active_detections = list(detections)

        frame_shape = None
        if frame is not None and hasattr(frame, "shape"):
            frame_shape = frame.shape[:2]

        return cls.analyze_relative_spatial_relations(
            target_class=target_class,
            detections=active_detections,
            frame_shape=frame_shape,
            tolerance_x=tolerance_x,
            tolerance_y=tolerance_y,
            anchor_config=anchor_config,
        )


# ===========================================================================
# Standalone Helper Functions & Dataclasses for Modular Spatial Reasoning
# ===========================================================================

@dataclass
class AnchorSelectionConfig:
    """
    Tunable parameters for selecting a reference/anchor object.
    Allows easy adjustment of selection criteria.
    """
    min_confidence: float = 0.30       # Minimum detection confidence for anchor
    weight_confidence: float = 0.35    # Weight for anchor detection confidence
    weight_distance: float = 0.45      # Weight for spatial proximity (closer is better)
    weight_prominence: float = 0.20    # Weight for anchor box area
    min_area: float = 400.0            # Minimum pixel area to filter tiny background noise


@dataclass
class RelativeSpatialResult:
    """Structured result of anchor-based relative spatial reasoning."""
    target_found: bool
    target_class: str
    target_count: int
    primary_target: DetectionItem | None
    all_targets: list[DetectionItem]
    anchor: DetectionItem | None
    direction: str | None
    target_centroid: tuple[float, float] | None
    anchor_centroid: tuple[float, float] | None
    distance_pixels: float | None
    description: str
    multiple_targets: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "target_found": self.target_found,
            "target_class": self.target_class,
            "target_count": self.target_count,
            "multiple_targets": self.multiple_targets,
            "primary_target": self.primary_target.model_dump() if self.primary_target else None,
            "anchor": self.anchor.model_dump() if self.anchor else None,
            "direction": self.direction,
            "target_centroid": self.target_centroid,
            "anchor_centroid": self.anchor_centroid,
            "distance_pixels": self.distance_pixels,
            "description": self.description,
        }


def calculate_centroid(bbox: BoundingBox | dict[str, float]) -> tuple[float, float]:
    """
    Calculates the 2D centroid of a bounding box.
    Formula:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
    """
    if isinstance(bbox, dict):
        x1 = bbox.get("x1", 0.0)
        y1 = bbox.get("y1", 0.0)
        x2 = bbox.get("x2", 0.0)
        y2 = bbox.get("y2", 0.0)
    else:
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (round(cx, 2), round(cy, 2))


def determine_relative_direction(
    target_center: tuple[float, float],
    anchor_center: tuple[float, float],
    tolerance_x: float = 25.0,
    tolerance_y: float = 25.0,
) -> str:
    """
    Compares the target centroid with the anchor centroid:
      - target center_x < anchor center_x - tolerance_x: target is left of anchor
      - target center_x > anchor center_x + tolerance_x: target is right of anchor
      - target center_y < anchor center_y - tolerance_y: target is above anchor
      - target center_y > anchor center_y + tolerance_y: target is below anchor

    Uses configurable tolerances (default 25px) so slight coordinate
    differences do not generate misleading directions.
    Combines horizontal and vertical relations (e.g. 'above and to the left of').
    """
    tx, ty = target_center
    ax, ay = anchor_center

    dx = tx - ax
    dy = ty - ay

    # Horizontal direction
    horiz = None
    if dx < -abs(tolerance_x):
        horiz = "left"
    elif dx > abs(tolerance_x):
        horiz = "right"

    # Vertical direction (Note: in image coordinates, smaller y is above)
    vert = None
    if dy < -abs(tolerance_y):
        vert = "above"
    elif dy > abs(tolerance_y):
        vert = "below"

    # Combine relations clearly
    if horiz and vert:
        return f"{vert} and to the {horiz} of"
    elif horiz:
        return f"to the {horiz} of"
    elif vert:
        return f"{vert}"
    else:
        return "right next to"


def select_anchor_object(
    target_item: DetectionItem,
    all_detections: Sequence[DetectionItem],
    target_class: str,
    config: AnchorSelectionConfig | None = None,
    frame_width: int = 640,
    frame_height: int = 480,
) -> DetectionItem | None:
    """
    Selects a suitable detected object as the reference/anchor object.
    Selection criteria:
      1. Strictly excludes the target object and any objects of the target class.
      2. Requires detection confidence >= min_confidence.
      3. Favors nearby, prominent objects using a composite score:
         score = w_conf * norm_conf + w_dist * (1 - norm_dist) + w_prom * norm_area
    """
    cfg = config or AnchorSelectionConfig()
    t_cx, t_cy = calculate_centroid(target_item.bbox)
    diag = max(math.sqrt(frame_width**2 + frame_height**2), 1.0)
    max_area = float(max(frame_width * frame_height, 1))

    target_name_clean = target_class.strip().lower()

    candidates: list[tuple[float, DetectionItem]] = []

    for det in all_detections:
        # Strictly exclude target class itself
        det_name = det.class_name.strip().lower()
        if det_name == target_name_clean:
            continue
        if det.class_id == target_item.class_id:
            continue

        # Confidence filter
        if det.confidence < cfg.min_confidence:
            continue

        # Size / area filter to avoid tiny noise
        area = det.bbox.width * det.bbox.height
        if area < cfg.min_area:
            continue

        # Proximity & geometry
        c_cx, c_cy = calculate_centroid(det.bbox)
        dist = math.sqrt((t_cx - c_cx) ** 2 + (t_cy - c_cy) ** 2)
        norm_dist = min(dist / diag, 1.0)
        norm_area = min(area / (max_area * 0.5), 1.0)
        norm_conf = min(max(det.confidence, 0.0), 1.0)

        # Composite score
        score = (
            cfg.weight_confidence * norm_conf
            + cfg.weight_distance * (1.0 - norm_dist)
            + cfg.weight_prominence * norm_area
        )
        candidates.append((score, det))

    if not candidates:
        return None

    # Pick candidate with the highest composite score
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def analyze_relative_spatial_relations(
    target_class: str,
    detections: Sequence[DetectionItem],
    frame_shape: tuple[int, int] | None = None,
    tolerance_x: float = 25.0,
    tolerance_y: float = 25.0,
    anchor_config: AnchorSelectionConfig | None = None,
) -> RelativeSpatialResult:
    """
    Analyzes spatial relationships between detected target object(s) and anchor objects.
    Produces natural language descriptions avoiding false 3D claims.
    """
    target_clean = target_class.strip().lower()
    fw = frame_shape[1] if frame_shape and len(frame_shape) >= 2 else 640
    fh = frame_shape[0] if frame_shape and len(frame_shape) >= 2 else 480

    # 1. Filter detections matching target class
    matched_targets: list[DetectionItem] = [
        d for d in detections
        if d.class_name.strip().lower() == target_clean
    ]

    # Case 1: Target not detected
    if not matched_targets:
        return RelativeSpatialResult(
            target_found=False,
            target_class=target_clean,
            target_count=0,
            primary_target=None,
            all_targets=[],
            anchor=None,
            direction=None,
            target_centroid=None,
            anchor_centroid=None,
            distance_pixels=None,
            description=f"I could not detect any {target_clean} in the image.",
            multiple_targets=False,
        )

    # Sort targets by confidence descending so primary is the clearest
    matched_targets.sort(key=lambda d: d.confidence, reverse=True)
    primary_target = matched_targets[0]
    target_count = len(matched_targets)
    is_multiple = target_count > 1

    t_cx, t_cy = calculate_centroid(primary_target.bbox)

    # 2. Select anchor object relative to primary target
    anchor = select_anchor_object(
        target_item=primary_target,
        all_detections=detections,
        target_class=target_clean,
        config=anchor_config,
        frame_width=fw,
        frame_height=fh,
    )

    # Case 2: Target found, but no suitable anchor object
    if anchor is None:
        if is_multiple:
            desc = f"I can see {target_count} {target_clean}s, but I could not find a suitable reference object."
        else:
            desc = f"I can see a {target_clean}, but I could not find a suitable reference object."

        return RelativeSpatialResult(
            target_found=True,
            target_class=target_clean,
            target_count=target_count,
            primary_target=primary_target,
            all_targets=matched_targets,
            anchor=None,
            direction=None,
            target_centroid=(t_cx, t_cy),
            anchor_centroid=None,
            distance_pixels=None,
            description=desc,
            multiple_targets=is_multiple,
        )

    # Case 3: Target and Anchor both found -> compute relative direction
    a_cx, a_cy = calculate_centroid(anchor.bbox)
    direction = determine_relative_direction(
        target_center=(t_cx, t_cy),
        anchor_center=(a_cx, a_cy),
        tolerance_x=tolerance_x,
        tolerance_y=tolerance_y,
    )
    dist_px = round(math.sqrt((t_cx - a_cx) ** 2 + (t_cy - a_cy) ** 2), 1)

    anchor_name = anchor.class_name.strip().lower()

    if is_multiple:
        desc = (
            f"I found {target_count} {target_clean}s. "
            f"The most prominent {target_clean} is {direction} the {anchor_name}."
        )
    else:
        desc = f"The {target_clean} is {direction} the {anchor_name}."

    return RelativeSpatialResult(
        target_found=True,
        target_class=target_clean,
        target_count=target_count,
        primary_target=primary_target,
        all_targets=matched_targets,
        anchor=anchor,
        direction=direction,
        target_centroid=(t_cx, t_cy),
        anchor_centroid=(a_cx, a_cy),
        distance_pixels=dist_px,
        description=desc,
        multiple_targets=is_multiple,
    )

