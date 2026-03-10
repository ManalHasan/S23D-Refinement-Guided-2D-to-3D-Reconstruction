"""
annotation_detector.py
-----------------------
Rule-based annotation detector using OpenCV.
Detects arrows, circles, X marks, and text labels from annotated sketches.
Maps detected annotations to one of 7 action types.
No LLM needed - zero cost, zero VRAM.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ActionType(Enum):
    ADD_OBJECT       = "Add a new object"
    CHANGE_SHAPE     = "Change the shape of an object"
    ADD_PATTERN      = "Add a pattern"
    REFERENCE_PATTERN= "Reference the pattern of another object"
    CHANGE_ORIENT    = "Change orientation of an object"
    CHANGE_SIZE      = "Change size of an object"
    MOVE_OBJECT      = "Move an object"
    GENERAL_EDIT     = "General edit"   # fallback


@dataclass
class DetectedAnnotation:
    action_type: ActionType
    confidence: float           # 0.0 - 1.0
    region_mask: np.ndarray     # binary mask of affected region
    description: str            # human readable description
    structured_prompt: str      # ready-to-use prompt for SD


class AnnotationDetector:
    """
    Detects visual annotations from a sketch image and maps them
    to structured prompts for Stable Diffusion + ControlNet.

    Color conventions (customize to your annotation style):
        RED   (#FF0000) -> arrows / movement / orientation
        GREEN (#00FF00) -> add object / new region
        BLUE  (#0000FF) -> resize / scale
        YELLOW(#FFFF00) -> pattern / texture reference
        BLACK / dark    -> part of original sketch (not annotation)
    """

    # HSV color ranges for annotation colors
    COLOR_RANGES = {
        "red": [
            (np.array([0,   150, 100]),  np.array([8,   255, 255])),
            (np.array([172, 150, 100]),  np.array([180, 255, 255])),
        ],
        "green":  [(np.array([40,  80,  60]),  np.array([90,  255, 255]))],
        "blue":   [(np.array([105, 80,  60]),  np.array([135, 255, 255]))],
        "yellow": [(np.array([15,  100, 100]), np.array([45,  255, 255]))],
    }

    def __init__(self, user_text: str = "", debug: bool = False):
        self.user_text  = user_text.strip().lower()
        self.debug      = debug

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def detect(
        self,
        annotated_sketch: np.ndarray,
        base_sketch: Optional[np.ndarray] = None
    ) -> DetectedAnnotation:
        """
        Main detection function.

        Parameters
        ----------
        annotated_sketch : BGR image with colored annotations drawn on top
        base_sketch      : optional clean sketch without annotations
                           (used to isolate annotation layer)

        Returns
        -------
        DetectedAnnotation with action type, mask, and structured prompt
        """
        # --- isolate annotation layer (all colors combined) ----------
        annotation_layer = self._isolate_annotations(
            annotated_sketch, base_sketch
        )

        # --- per-color layers for reliable element detection ----------
        red_layer    = self._isolate_color(annotated_sketch, "red")
        green_layer  = self._isolate_color(annotated_sketch, "green")
        blue_layer   = self._isolate_color(annotated_sketch, "blue")
        yellow_layer = self._isolate_color(annotated_sketch, "yellow")

        # --- detect individual annotation elements --------------------
        # arrows: check red + any color layer
        has_arrow_red    = (
            self._detect_arrows(red_layer) or
            self._detect_arrows(annotation_layer)
        )
        has_arrow        = has_arrow_red
        has_circle_blue  = self._detect_circles(blue_layer)
        has_circle       = has_circle_blue or self._detect_circles(annotation_layer)
        has_x_mark       = self._detect_x_marks(annotation_layer)
        has_scribble_grn = self._detect_scribbles(green_layer)
        has_scribble_ylw = self._detect_scribbles(yellow_layer)
        has_scribble     = (
            has_scribble_grn or has_scribble_ylw or
            self._detect_scribbles(annotation_layer)
        )
        dominant_color   = self._dominant_annotation_color(annotation_layer)

        # refine dominant color using per-layer pixel counts
        color_counts = {
            "red":    cv2.countNonZero(cv2.cvtColor(red_layer,    cv2.COLOR_BGR2GRAY)),
            "green":  cv2.countNonZero(cv2.cvtColor(green_layer,  cv2.COLOR_BGR2GRAY)),
            "blue":   cv2.countNonZero(cv2.cvtColor(blue_layer,   cv2.COLOR_BGR2GRAY)),
            "yellow": cv2.countNonZero(cv2.cvtColor(yellow_layer, cv2.COLOR_BGR2GRAY)),
        }
        if max(color_counts.values()) > 50:
            dominant_color = max(color_counts, key=color_counts.get)
        else:
            dominant_color = "none"

        # override has_circle_blue if blue pixels exist but failed HoughCircles
        if color_counts["blue"] > 100:
            has_circle_blue = True
            has_circle = True

        # override has_scribble_ylw
        if color_counts["yellow"] > 80:
            has_scribble_ylw = True
            has_scribble = True

        if self.debug:
            print(f"[Detector] arrow={has_arrow}, circle={has_circle}({has_circle_blue}), "
                  f"x_mark={has_x_mark}, scribble={has_scribble}, "
                  f"color={dominant_color}, counts={color_counts}")

        region_mask      = self._build_region_mask(annotation_layer)

        # --- classify action type -------------------------------------
        action, confidence = self._classify_action(
            has_arrow, has_circle, has_x_mark,
            has_scribble, dominant_color,
            has_scribble_ylw=has_scribble_ylw,
        )

        # --- build structured prompt ----------------------------------
        prompt = self._build_prompt(action, dominant_color)

        return DetectedAnnotation(
            action_type      = action,
            confidence       = confidence,
            region_mask      = region_mask,
            description      = self._describe(action, dominant_color),
            structured_prompt= prompt,
        )

    # ------------------------------------------------------------------
    # Annotation isolation
    # ------------------------------------------------------------------
    def _isolate_annotations(
        self,
        annotated: np.ndarray,
        base: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        If we have a clean base sketch, subtract it to get pure annotations.
        Otherwise detect colored pixels that stand out from sketch lines.
        """
        if base is not None:
            diff = cv2.absdiff(annotated, base)
            _, mask = cv2.threshold(
                cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY),
                25, 255, cv2.THRESH_BINARY
            )
            annotation_layer = cv2.bitwise_and(
                annotated, annotated, mask=mask
            )
            return annotation_layer

        # PIL → numpy gives RGB; convert to BGR for OpenCV HSV
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Isolate each color separately for cleaner detection
        # Isolate each color separately for cleaner detection
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_name, ranges in self.COLOR_RANGES.items():
            for lo, hi in ranges:
                color_mask = cv2.bitwise_or(
                    color_mask, cv2.inRange(hsv, lo, hi)
                )
        annotation_layer = cv2.bitwise_and(
            annotated, annotated, mask=color_mask
        )
        return annotation_layer

    def _isolate_color(
        self, annotated: np.ndarray, color_name: str
    ) -> np.ndarray:
        """Return annotation layer containing only a specific color."""
        bgr  = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.COLOR_RANGES.get(color_name, []):
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return cv2.bitwise_and(annotated, annotated, mask=mask)

    # ------------------------------------------------------------------
    # Element detectors
    # ------------------------------------------------------------------
    def _detect_arrows(self, layer: np.ndarray) -> bool:
        """
        Detect arrow-like shapes using two strategies:
        1. Contour solidity + aspect ratio (for filled arrows)
        2. Long line segments via HoughLinesP (for line-based arrows)
        """
        gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # Strategy 1: contour shape analysis
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            hull   = cv2.convexHull(cnt)
            hull_a = cv2.contourArea(hull)
            if hull_a == 0:
                continue
            solidity = area / hull_a
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 1)
            if 0.3 < solidity < 0.90 and aspect > 1.8:
                return True

        # Strategy 2: long line segments (works even for thin drawn arrows)
        edges = cv2.Canny(binary, 30, 100)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=20, minLineLength=30, maxLineGap=15,
        )
        if lines is not None and len(lines) >= 1:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 40:
                    return True

        return False

    def _detect_circles(self, layer: np.ndarray) -> bool:
        """Detect circular annotations using HoughCircles."""
        gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=25,
            minRadius=15,
            maxRadius=200,
        )
        return circles is not None

    def _detect_x_marks(self, layer: np.ndarray) -> bool:
        """
        Detect X marks by looking for two crossing line segments.
        Uses line detection + intersection analysis.
        """
        gray    = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 50, 150)
        lines   = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10,
        )
        if lines is None or len(lines) < 2:
            return False

        # check for crossing lines at ~45 degree angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)

        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs(angles[i] - angles[j])
                # two lines roughly perpendicular or at ~90 degrees
                if 70 < diff < 110:
                    return True
        return False

    def _detect_scribbles(self, layer: np.ndarray) -> bool:
        """Detect scribble marks (high curvature, dense strokes)."""
        gray    = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for cnt in contours:
            if len(cnt) < 20:
                continue
            area    = cv2.contourArea(cnt)
            length  = cv2.arcLength(cnt, closed=False)
            if area == 0:
                continue
            # scribbles: high perimeter-to-area ratio
            compactness = (length ** 2) / (4 * np.pi * area + 1)
            if compactness > 10:
                return True
        return False

    def _dominant_annotation_color(self, layer: np.ndarray) -> str:
        """Find the dominant non-black annotation color."""
        bgr    = cv2.cvtColor(layer, cv2.COLOR_RGB2BGR)
        hsv    = cv2.cvtColor(bgr,   cv2.COLOR_BGR2HSV)
        scores  = {}
        for color_name, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
            scores[color_name] = cv2.countNonZero(mask)

        if max(scores.values()) < 50:   # no strong color found
            return "none"
        return max(scores, key=scores.get)

    def _build_region_mask(self, layer: np.ndarray) -> np.ndarray:
        """Build a binary mask of the annotated region (dilated)."""
        gray    = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        return dilated

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify_action(
        self,
        has_arrow: bool,
        has_circle: bool,
        has_x_mark: bool,
        has_scribble: bool,
        color: str,
        has_scribble_ylw: bool = False,
    ):
        """
        Decision tree mapping detected elements to action types.
        Returns (ActionType, confidence_float).

        Priority rules (highest to lowest):
        1. X mark                          -> general edit (remove)
        2. Arrow + text "pattern"          -> reference pattern
        3. Scribble + yellow color         -> add pattern
        4. Arrow + red color               -> move object
        5. Arrow alone                     -> move or orient
        6. Circle + blue                   -> resize
        7. Scribble + green                -> add new object
        8. Circle alone                    -> modify shape
        9. Scribble alone                  -> add pattern / shape
        10. Text keyword fallback
        """
        text = self.user_text

        # highest priority: blue circle = resize (regardless of x_mark)
        if has_circle and color == "blue":
            return ActionType.CHANGE_SIZE, 0.90

        if has_x_mark and not has_circle:
            return ActionType.GENERAL_EDIT, 0.85

        if has_arrow and "pattern" in text:
            return ActionType.REFERENCE_PATTERN, 0.90

        if (has_scribble_ylw or color == "yellow") and not has_arrow:
            return ActionType.ADD_PATTERN, 0.88

        if has_arrow and color == "red":
            if any(k in text for k in ["rotat", "turn", "orient", "flip"]):
                return ActionType.CHANGE_ORIENT, 0.85
            return ActionType.MOVE_OBJECT, 0.87

        if has_arrow and not has_circle:
            if any(k in text for k in ["rotat", "turn", "orient", "flip"]):
                return ActionType.CHANGE_ORIENT, 0.80
            return ActionType.MOVE_OBJECT, 0.78

        if has_circle and color == "blue":
            return ActionType.CHANGE_SIZE, 0.85

        if has_circle and any(
            k in text for k in ["resize","bigger","smaller","scale","size"]
        ):
            return ActionType.CHANGE_SIZE, 0.80

        if has_scribble and color == "green":
            return ActionType.ADD_OBJECT, 0.82

        if has_circle and not has_arrow:
            if any(k in text for k in ["shape", "reshape", "form"]):
                return ActionType.CHANGE_SHAPE, 0.75
            return ActionType.CHANGE_SHAPE, 0.65

        if has_scribble:
            if any(k in text for k in ["pattern", "texture", "stripe"]):
                return ActionType.ADD_PATTERN, 0.75
            return ActionType.ADD_PATTERN, 0.60

        # text-only keyword fallback
        keyword_map = {
            ActionType.MOVE_OBJECT:        ["move", "shift", "place", "relocate"],
            ActionType.ADD_OBJECT:         ["add", "insert", "put", "place a"],
            ActionType.CHANGE_SIZE:        ["resize", "bigger", "smaller", "scale"],
            ActionType.CHANGE_ORIENT:      ["rotate", "flip", "turn", "orient"],
            ActionType.ADD_PATTERN:        ["pattern", "texture", "stripe", "polka"],
            ActionType.REFERENCE_PATTERN:  ["reference", "same as", "like the"],
            ActionType.CHANGE_SHAPE:       ["shape", "reshape", "form", "curve"],
        }
        for action, keywords in keyword_map.items():
            if any(k in text for k in keywords):
                return action, 0.65

        return ActionType.GENERAL_EDIT, 0.50

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _build_prompt(self, action: ActionType, color: str) -> str:
        """
        Build a structured natural language prompt for SD + ControlNet.
        Incorporates user_text for additional semantic detail.
        """
        detail = f" {self.user_text}" if self.user_text else ""

        action_guide = {
            ActionType.ADD_OBJECT:
                f"Add a new object to the scene{detail}. "
                "Keep the rest of the image unchanged. "
                "High quality 2D illustration, detailed, professional artwork.",

            ActionType.CHANGE_SHAPE:
                f"Reshape the indicated object{detail}. "
                "Maintain position and surrounding elements. "
                "High quality 2D illustration.",

            ActionType.ADD_PATTERN:
                f"Add a detailed pattern or texture to the marked surface{detail}. "
                "Keep surrounding areas unchanged. "
                "High quality 2D illustration.",

            ActionType.REFERENCE_PATTERN:
                f"Apply the pattern or texture from the referenced element{detail}. "
                "Maintain object positions. High quality 2D illustration.",

            ActionType.CHANGE_ORIENT:
                f"Rotate or reorient the indicated object{detail}. "
                "Keep position and scale the same. "
                "High quality 2D illustration.",

            ActionType.CHANGE_SIZE:
                f"Resize the indicated object{detail}. "
                "Keep position and style consistent. "
                "High quality 2D illustration.",

            ActionType.MOVE_OBJECT:
                f"Move the indicated object to the new position shown{detail}. "
                "Keep all other elements unchanged. "
                "High quality 2D illustration.",

            ActionType.GENERAL_EDIT:
                f"Edit the image as indicated by the annotations{detail}. "
                "High quality 2D illustration, detailed, professional artwork.",
        }
        return action_guide[action]

    def _describe(self, action: ActionType, color: str) -> str:
        return (
            f"Detected: {action.value} "
            f"(annotation color: {color}, user text: '{self.user_text}')"
        )
