from typing import Any
import cv2
import numpy as np

# Deterministic color palette for distinct classes
np.random.seed(42)
COLOR_PALETTE = [
    tuple(int(c) for c in color)
    for color in np.random.randint(50, 255, size=(100, 3))
]


class Visualizer:
    """Renders high-visibility bounding boxes, badges, labels, and statistics."""

    @staticmethod
    def detections_to_telemetry(detections, frame_shape: tuple[int, ...]) -> list[dict]:
        """Convert a list of DetectionItem objects into spatial telemetry dicts."""
        h_frame, w_frame = frame_shape[:2]
        telemetry_items = []
        for d in detections:
            cx = (d.bbox.x1 + d.bbox.x2) / 2.0
            norm_x = (cx - w_frame / 2.0) / (w_frame / 2.0)
            azimuth = round(norm_x * 34.0, 1)
            if azimuth < -10:
                sector = f"Left ({azimuth}°)"
            elif azimuth > 10:
                sector = f"Right (+{azimuth}°)"
            else:
                sector = f"Center ({azimuth:+}°)"

            h_box = max(d.bbox.height, 10)
            dist_m = round(max(0.4, min(3.5, 140.0 / h_box)), 1)
            dist_ft = round(dist_m * 3.28084, 1)

            conf_val = d.confidence * 100.0 if d.confidence <= 1.0 else d.confidence

            telemetry_items.append({
                "class_name": d.class_name,
                "class_id": d.class_id,
                "confidence": round(conf_val, 1),
                "azimuth_deg": azimuth,
                "sector": sector,
                "distance_m": dist_m,
                "distance_ft": dist_ft,
                "bbox": {
                    "x1": d.bbox.x1,
                    "y1": d.bbox.y1,
                    "x2": d.bbox.x2,
                    "y2": d.bbox.y2,
                },
            })
        return telemetry_items

    @staticmethod
    def get_class_color(class_id: int) -> tuple[int, int, int]:
        """Return a deterministic BGR color tuple for a given class ID."""
        return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

    @classmethod
    def draw_corner_brackets(
        cls,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int] = (0, 255, 128),
        length: int = 16,
        thickness: int = 3,
    ) -> None:
        """Draw 4 corner reticle brackets around a target bounding box."""
        # Top-left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    @classmethod
    def draw_detections(
        cls,
        frame: np.ndarray,
        results,
        class_names: dict[int, str],
        target_ids: set[int] | list[int] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and class labels with filled contrast badges on an image frame.
        If target_ids is provided, ONLY detections matching target_ids are drawn.
        """
        annotated = frame.copy()
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return annotated

        h_frame, w_frame = annotated.shape[:2]
        target_ids_set = set(target_ids) if target_ids is not None else None

        for box in boxes:
            cls_id = int(box.cls[0])
            if target_ids_set is not None and cls_id not in target_ids_set:
                # Filter out all non-target objects
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_name = class_names.get(cls_id, f"Class {cls_id}")

            color = (0, 255, 128) if target_ids_set else cls.get_class_color(cls_id)

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
            if target_ids_set:
                cls.draw_corner_brackets(annotated, x1, y1, x2, y2, color=(0, 255, 255), length=18, thickness=3)

            # Label text
            label = f"TARGET: {cls_name.upper()} {int(conf * 100)}%" if target_ids_set else f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thickness = 1

            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Keep badge inside frame bounds
            if y1 - text_h - 8 >= 0:
                badge_y1 = y1 - text_h - 8
                badge_y2 = y1
                text_y = y1 - 4
            else:
                badge_y1 = y1
                badge_y2 = min(y1 + text_h + 8, h_frame - 1)
                text_y = y1 + text_h + 4

            badge_x1 = x1
            badge_x2 = min(x1 + text_w + 10, w_frame - 1)
            text_x = x1 + 5

            # Draw filled label background
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 23, 42), -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 220) if target_ids_set else (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

    @classmethod
    def draw_target_overlay(
        cls,
        frame: np.ndarray,
        target_items: list[dict],
        active_targets: list[str],
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw an exclusive target-lock HUD overlay showing ONLY target objects.
        """
        annotated = frame.copy()
        h_frame, w_frame = annotated.shape[:2]

        for item in target_items:
            bbox = item["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            conf = item["confidence"]
            cls_name = item["class_name"]
            sector = item.get("sector", "")
            dist_ft = item.get("distance_ft", 0.0)

            # Draw neon target rectangle
            target_color = (0, 240, 144)  # Emerald green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), target_color, thickness=2)

            # Draw corner reticle brackets
            cls.draw_corner_brackets(annotated, x1, y1, x2, y2, color=(0, 229, 255), length=20, thickness=3)

            # Center target crosshair
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(annotated, (cx, cy), (0, 229, 255), cv2.MARKER_CROSS, 16, 2)

            # Target badge
            label = f"{cls_name.title()} (~{dist_ft:.1f} ft)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x2 = min(w_frame - 1, x1 + text_w + 14)

            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), (15, 23, 42), -1)
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), target_color, 1)
            cv2.putText(
                annotated,
                label,
                (x1 + 6, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Top Humanized Status Banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        if target_items:
            best = target_items[0]
            sector_raw = best.get("sector", "")
            if "Left" in sector_raw:
                dir_str = "on your left"
            elif "Right" in sector_raw:
                dir_str = "on your right"
            else:
                dir_str = "in front of you"
            top_msg = f"Found: {best['class_name'].title()} ({dir_str}, ~{best['distance_ft']} ft away)"
            banner_bg = (15, 23, 42)
            banner_border = (0, 240, 144)
            text_color = (255, 255, 255)
        else:
            targets_str = ", ".join(t.title() for t in active_targets) if active_targets else "object"
            top_msg = f"Looking for {targets_str}..."
            banner_bg = (15, 23, 42)
            banner_border = (56, 189, 248)
            text_color = (226, 232, 240)

        (tw, th), _ = cv2.getTextSize(top_msg, font, 0.52, 1)
        cv2.rectangle(annotated, (12, 12), (12 + tw + 20, 12 + th + 14), banner_bg, -1)
        cv2.rectangle(annotated, (12, 12), (12 + tw + 20, 12 + th + 14), banner_border, 1)
        cv2.putText(annotated, top_msg, (22, 12 + th + 4), font, 0.52, text_color, 1, cv2.LINE_AA)

        return annotated

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float = 0.0, count: int = 0, summary: str = "") -> np.ndarray:
        """Draw a clean, simple detection status badge in the top-left corner."""
        annotated = frame.copy()
        if count > 0:
            text = f"Detected ({count}): {summary}" if summary else f"{count} objects in view"
        else:
            text = "Scanning surroundings..."

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        font_thickness = 1

        (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(annotated, (12, 12), (12 + w + 16, 12 + h + 14), (15, 23, 42), -1)
        cv2.rectangle(annotated, (12, 12), (12 + w + 16, 12 + h + 14), (56, 189, 248), 1)
        cv2.putText(
            annotated,
            text,
            (20, 12 + h + 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
        return annotated

    @classmethod
    def draw_spatial_relation_overlay(
        cls,
        frame: np.ndarray,
        spatial_result: Any,
    ) -> np.ndarray:
        """
        Draws high-visibility visualization for anchor-based spatial reasoning:
          - Target bounding box & reticle (bright emerald)
          - Anchor bounding box & reticle (amber / gold)
          - Connecting vector between centroids
          - Top HUD banner showing natural-language spatial description
        """
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw Target(s)
        if spatial_result.primary_target:
            t = spatial_result.primary_target
            tx1, ty1 = int(t.bbox.x1), int(t.bbox.y1)
            tx2, ty2 = int(t.bbox.x2), int(t.bbox.y2)
            t_color = (0, 255, 128)  # Neon Emerald

            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), t_color, 2)
            cls.draw_corner_brackets(annotated, tx1, ty1, tx2, ty2, color=t_color, length=18, thickness=3)

            t_label = f"TARGET: {t.class_name.upper()} ({int(t.confidence * 100)}%)"
            (lw, lh), _ = cv2.getTextSize(t_label, font, 0.48, 1)
            cv2.rectangle(annotated, (tx1, max(0, ty1 - 22)), (tx1 + lw + 12, max(0, ty1)), (15, 23, 42), -1)
            cv2.putText(annotated, t_label, (tx1 + 6, max(0, ty1 - 7)), font, 0.48, t_color, 1, cv2.LINE_AA)

            # Draw target centroid dot
            if spatial_result.target_centroid:
                tcx, tcy = int(spatial_result.target_centroid[0]), int(spatial_result.target_centroid[1])
                cv2.circle(annotated, (tcx, tcy), 5, t_color, -1)
                cv2.circle(annotated, (tcx, tcy), 8, (255, 255, 255), 1)

        # 2. Draw Anchor (if selected)
        if spatial_result.anchor:
            a = spatial_result.anchor
            ax1, ay1 = int(a.bbox.x1), int(a.bbox.y1)
            ax2, ay2 = int(a.bbox.x2), int(a.bbox.y2)
            a_color = (0, 191, 255)  # Amber / Gold

            cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), a_color, 2)
            cls.draw_corner_brackets(annotated, ax1, ay1, ax2, ay2, color=a_color, length=18, thickness=3)

            a_label = f"ANCHOR: {a.class_name.upper()} ({int(a.confidence * 100)}%)"
            (lw, lh), _ = cv2.getTextSize(a_label, font, 0.48, 1)
            cv2.rectangle(annotated, (ax1, max(0, ay1 - 22)), (ax1 + lw + 12, max(0, ay1)), (15, 23, 42), -1)
            cv2.putText(annotated, a_label, (ax1 + 6, max(0, ay1 - 7)), font, 0.48, a_color, 1, cv2.LINE_AA)

            # Draw anchor centroid dot
            if spatial_result.anchor_centroid:
                acx, acy = int(spatial_result.anchor_centroid[0]), int(spatial_result.anchor_centroid[1])
                cv2.circle(annotated, (acx, acy), 5, a_color, -1)
                cv2.circle(annotated, (acx, acy), 8, (255, 255, 255), 1)

                # Draw connecting line between centroids
                if spatial_result.target_centroid:
                    tcx, tcy = int(spatial_result.target_centroid[0]), int(spatial_result.target_centroid[1])
                    cv2.line(annotated, (tcx, tcy), (acx, acy), (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Top HUD Banner with Description
        desc_text = spatial_result.description
        (dw, dh), _ = cv2.getTextSize(desc_text, font, 0.52, 1)
        banner_w = max(dw + 28, 300)
        cv2.rectangle(annotated, (12, 12), (12 + banner_w, 12 + dh + 16), (15, 23, 42), -1)
        border_col = (0, 255, 128) if spatial_result.target_found else (239, 68, 68)
        cv2.rectangle(annotated, (12, 12), (12 + banner_w, 12 + dh + 16), border_col, 1)
        cv2.putText(annotated, desc_text, (22, 12 + dh + 5), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
