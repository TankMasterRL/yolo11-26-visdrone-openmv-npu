"""
region_counter.py — Region-based object counting for Luxonis OAK / OAK4
========================================================================

Host-side NumPy / OpenCV port of openmv-scripts/region_counter.py. Public
API and semantics are intentionally identical so the OAK pipeline produces
the same per-frame counts and printout format as the OpenMV scripts:

    counts -> {"Zone-A": 3, "Zone-B": 1, ...}

Usage:
    from region_counter import RegionCounter
    counter = RegionCounter([
        {"name": "Left",  "rect": (0,   0, 160, 240), "color": (0,   0, 255)},
        {"name": "Right", "rect": (160, 0, 160, 240), "color": (0, 255,   0)},
    ])
    counts = counter.update(detections, frame_bgr, draw=True)

Notes
-----
- Region rectangles are in pixel coordinates of the frame passed to
  ``update``; choose dimensions matching that frame.
- Detection objects are duck-typed: ``.rect()`` -> (x, y, w, h),
  ``.score()`` -> float, ``.class_id()`` -> int. The DepthAI pipeline
  in ``_pipeline.py`` adapts ``dai.ImgDetection`` into that shape.
- ``color`` tuples follow OpenCV BGR order. The OpenMV scripts use RGB;
  swap accordingly when porting region defs.
"""

from __future__ import annotations

import cv2


class RegionCounter:
    """Track per-region object counts from YOLO detections."""

    def __init__(self, regions, min_score=0.35, target_classes=None):
        """
        Parameters
        ----------
        regions : list[dict]
            Each dict must have:
              - "name"  : str           – display label
              - "rect"  : (x, y, w, h)  – bounding region in image coords
              - "color" : (b, g, r)     – drawing colour (OpenCV BGR)
        min_score : float
            Minimum detection confidence to count.
        target_classes : list[int] | None
            If set, only count objects whose class index is in this list.
            None means count all classes.
        """
        self._regions = regions
        self._min_score = min_score
        self._target_classes = set(target_classes) if target_classes else None
        self._rects = []
        for r in regions:
            x, y, w, h = r["rect"]
            self._rects.append((x, y, x + w, y + h))

    @staticmethod
    def _centre(det):
        x, y, w, h = det.rect()
        return x + w // 2, y + h // 2

    def update(self, detections, img=None, draw=True):
        """Process one frame of detections, return ``{name: count}``."""
        counts = {r["name"]: 0 for r in self._regions}

        for det in detections:
            if det.score() < self._min_score:
                continue

            if self._target_classes is not None:
                cl = det.class_id()
                if cl not in self._target_classes:
                    continue

            cx, cy = self._centre(det)

            if draw and img is not None:
                x, y, w, h = det.rect()
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)

            for i, (x0, y0, x1, y1) in enumerate(self._rects):
                if x0 <= cx < x1 and y0 <= cy < y1:
                    counts[self._regions[i]["name"]] += 1
                    if draw and img is not None:
                        cv2.drawMarker(
                            img, (cx, cy),
                            self._regions[i]["color"],
                            markerType=cv2.MARKER_CROSS,
                            markerSize=8, thickness=2,
                        )
                    break

        if draw and img is not None:
            for r in self._regions:
                rx, ry, rw, rh = r["rect"]
                cv2.rectangle(
                    img, (rx, ry), (rx + rw, ry + rh),
                    r["color"], 2,
                )
                label = "%s:%d" % (r["name"], counts[r["name"]])
                cv2.putText(
                    img, label, (rx + 4, ry + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    r["color"], 1, cv2.LINE_AA,
                )

        return counts

    @staticmethod
    def total(counts):
        return sum(counts.values())
