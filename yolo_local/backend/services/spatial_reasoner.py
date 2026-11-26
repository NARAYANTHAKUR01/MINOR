"""
Spatial Reasoning & Location Response Generator.

Calculates normalized 3x3 frame coordinates for detected bounding boxes:
- Grid layout:
    top-left    | top    | top-right
    left        | center | right
    bottom-left | bottom | bottom-right

Matches requested regions (broad or specific) and formats natural-language spoken feedback.
"""

from typing import Any
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
