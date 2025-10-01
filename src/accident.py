import math
from typing import Dict, Tuple, List


BBox = Tuple[int, int, int, int]


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    a1 = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    a2 = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


class AccidentDetector:
    """Simple collision heuristic based on bbox overlap and low motion.

    - Maintains per-object recent y-centroid positions to estimate motion.
    - Flags accident if any pair of boxes overlaps above iou_threshold and
      both objects' per-frame motion magnitude is below dy_threshold for
      at least consecutive_frames frames.
    """

    def __init__(self, iou_threshold: float = 0.2, dy_threshold: float = 2.0, consecutive_frames: int = 3):
        self.iou_threshold = iou_threshold
        self.dy_threshold = dy_threshold
        self.consecutive_frames = consecutive_frames
        self.object_tracks: Dict[int, List[int]] = {}
        self.stationary_counts: Dict[Tuple[int, int], int] = {}

    def update(self, oid_to_bbox: Dict[int, BBox], oid_to_centroid_y: Dict[int, int]) -> bool:
        # update tracks and estimate per-object dy (speed proxy)
        oid_to_dy: Dict[int, float] = {}
        for oid, cy in oid_to_centroid_y.items():
            if oid not in self.object_tracks:
                self.object_tracks[oid] = []
            self.object_tracks[oid].append(cy)
            if len(self.object_tracks[oid]) > 10:
                self.object_tracks[oid] = self.object_tracks[oid][-10:]
            if len(self.object_tracks[oid]) >= 2:
                oid_to_dy[oid] = abs(self.object_tracks[oid][-1] - self.object_tracks[oid][-2])
            else:
                oid_to_dy[oid] = 0.0

        # check pairs for overlap + low motion, count consecutive frames
        oids = list(oid_to_bbox.keys())
        accident_detected = False
        for i in range(len(oids)):
            for j in range(i + 1, len(oids)):
                oi = oids[i]
                oj = oids[j]
                iou = compute_iou(oid_to_bbox[oi], oid_to_bbox[oj])
                low_motion = (oid_to_dy.get(oi, 0.0) < self.dy_threshold) and (oid_to_dy.get(oj, 0.0) < self.dy_threshold)
                key = (min(oi, oj), max(oi, oj))
                if iou > self.iou_threshold and low_motion:
                    self.stationary_counts[key] = self.stationary_counts.get(key, 0) + 1
                else:
                    self.stationary_counts[key] = 0
                if self.stationary_counts[key] >= self.consecutive_frames:
                    accident_detected = True
        return accident_detected


