"""
region_counter.py — Region-based object counting for OpenMV
=============================================================

Counts how many detected objects have their centre inside one or more
user-defined rectangular regions.  Works with any YOLO model loaded
via the ``ml`` module with ``ultralytics.YOLO`` postprocessing.

Upload this file alongside your main script to the OpenMV Cam.

Usage in main script:
    from region_counter import RegionCounter

    regions = [
        {"name": "Zone-A", "rect": (0, 0, 160, 120),  "color": (255, 0, 0)},
        {"name": "Zone-B", "rect": (160, 0, 160, 120), "color": (0, 255, 0)},
    ]
    counter = RegionCounter(regions)

    # Each frame:
    counts = counter.update(detections, img)
    #  counts = {"Zone-A": 3, "Zone-B": 1, ...}
"""

from micropython import const

_FONT_SCALE = const(1)
_FONT_COLOR = const(255)  # white for RGB565


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
              - "color" : (r, g, b)     – drawing colour
        min_score : float
            Minimum detection confidence to count.
        target_classes : list[int] | None
            If set, only count objects whose class index is in this list.
            None means count all classes.
        """
        self._regions = regions
        self._min_score = min_score
        self._target_classes = set(target_classes) if target_classes else None
        # Pre-build lookup for fast point-in-rect tests
        self._rects = []
        for r in regions:
            x, y, w, h = r["rect"]
            self._rects.append((x, y, x + w, y + h))

    # ------------------------------------------------------------------
    @staticmethod
    def _centre(det):
        """Extract centre (cx, cy) from an ultralytics detection object."""
        r = det.rect()
        return r[0] + r[2] // 2, r[1] + r[3] // 2

    # ------------------------------------------------------------------
    def update(self, detections, img=None, draw=True):
        """
        Process one frame of detections.

        Parameters
        ----------
        detections : list
            Output of ``model.predict([img])`` with YOLO postprocessor.
        img : image.Image | None
            If provided and draw=True, overlay regions + counts.
        draw : bool
            Whether to draw on img.

        Returns
        -------
        dict[str, int]  – per-region object counts.
        """
        # Initialise counts
        counts = {}
        for r in self._regions:
            counts[r["name"]] = 0

        for det in detections:
            # Score filter
            if det.score() < self._min_score:
                continue

            # Class filter
            if self._target_classes is not None:
                cl = det.class_id() if hasattr(det, "class_id") else det.label()
                if cl not in self._target_classes:
                    continue

            cx, cy = self._centre(det)

            # Draw detection box
            if draw and img is not None:
                img.draw_rectangle(det.rect(), color=(255, 255, 255))

            # Check each region
            for i, (x0, y0, x1, y1) in enumerate(self._rects):
                if x0 <= cx < x1 and y0 <= cy < y1:
                    counts[self._regions[i]["name"]] += 1
                    # Coloured cross for matched detections
                    if draw and img is not None:
                        img.draw_cross(
                            cx, cy, size=4,
                            color=self._regions[i]["color"]
                        )
                    break  # each detection counted once

        # Draw region overlays and counts
        if draw and img is not None:
            for i, r in enumerate(self._regions):
                rx, ry, rw, rh = r["rect"]
                img.draw_rectangle(rx, ry, rw, rh, color=r["color"], thickness=2)
                label = "%s:%d" % (r["name"], counts[r["name"]])
                img.draw_string(
                    rx + 2, ry + 2, label,
                    color=r["color"], scale=_FONT_SCALE,
                )

        return counts

    # ------------------------------------------------------------------
    def total(self, counts):
        """Sum all region counts."""
        t = 0
        for v in counts.values():
            t += v
        return t
