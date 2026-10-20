"""Camera-based object detection from raw pixels.

Pipeline for one frame (analyse_frame):
1. BGR -> HSV, then a colour mask per supported colour (hue band plus
   minimum saturation and value), cleaned with a 3x3 open/close.
2. External contours of each mask; each blob gets shape measurements.
3. Rules that reject what cannot be the target:
   - area below MIN_AREA_PX                  -> noise
   - top of the blob below the horizon row   -> flat floor marking
     (the coloured pads; the red meeting-room pad has the ball's hue)
   - touching the image border               -> partial, shape unreliable
4. Shape classification of what remains:
   - "round":   bounding-box fill 0.62-0.88 (disc = 0.785), enclosing-
                circle fill >= 0.70, aspect 0.7-1.4         -> ball
   - "upright": bounding-box fill >= 0.88 or circle fill < 0.70
                                                     -> box or cylinder
   - otherwise "unclear".
5. Confidence for a round blob: the mean of how well its three shape
   measures match an ideal disc, scaled down for small blobs.

Webots' built-in Recognition is not used: the world's camera has no
Recognition node, and the specification asks for image analysis.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# Camera geometry (E-puck.proto camera_fieldOfView 0.84 rad; the world sets
# 320 x 240). Focal length in pixels = (width / 2) / tan(fov / 2).
CAMERA_FOV_RAD = 0.84
BALL_DIAMETER_M = 0.18

# OpenCV HSV: H 0-179, S and V 0-255. Red wraps around 0. Measured colours
# of world objects: red ball H=0 S=242, red box H=0 S=237, red meeting pad
# H=0 S=188, orange office-A pad H=14, brown furniture H=13-15, yellow
# cylinder H=27, green ball H=64, blue box/cylinder H~113, reception pad
# H=106, purple pad H=135. Bands leave a gap before each neighbour.
# Red needs S >= 160: in rendered Webots frames the ball and red box measure
# S 174-196 (median), while the red meeting-room pad measures at most 154.
# With a lower limit the pad under the robot merged with the ball into one
# blob and the ball was missed (run 2026-10-20 16:02, meeting_east spawn).
COLOUR_RANGES = {
    "red": [((0, 160, 40), (8, 255, 255)), ((172, 160, 40), (179, 255, 255))],
    "yellow": [((20, 120, 60), (34, 255, 255))],
    "green": [((45, 100, 40), (85, 255, 255))],
    "blue": [((100, 120, 40), (128, 255, 255))],
}

MIN_AREA_PX = 30                 # about a 6 px disc; smaller is noise
MIN_MEASURABLE_DIAMETER_PX = 10  # below this the shape cannot be trusted
HORIZON_MARGIN_PX = 3            # top must be this far above the centre row
BORDER_MARGIN_PX = 1

ROUND_EXTENT = (0.62, 0.88)      # area / bounding-box area; disc = pi/4
ROUND_CIRCLE_FILL_MIN = 0.70     # area / enclosing-circle area; square = 0.64
ROUND_ASPECT = (0.7, 1.4)        # width / height
UPRIGHT_EXTENT_MIN = 0.88


@dataclass
class Detection:
    object_type: str            # "ball", "upright" (box/cylinder) or "unclear"
    colour: str
    confidence: float
    image_x: Optional[float] = None
    image_y: Optional[float] = None
    diameter_px: float = 0.0
    width_px: int = 0
    height_px: int = 0
    area_px: float = 0.0
    extent: float = 0.0
    circle_fill: float = 0.0
    aspect: float = 0.0
    bearing_rad: float = 0.0    # + = left of the camera axis
    distance_m: Optional[float] = None
    rejected: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self):
        return self.rejected is None

    def summary(self):
        return {
            "type": self.object_type, "colour": self.colour,
            "confidence": round(self.confidence, 2),
            "u": None if self.image_x is None else round(self.image_x, 1),
            "v": None if self.image_y is None else round(self.image_y, 1),
            "diameter_px": round(self.diameter_px, 1),
            "extent": round(self.extent, 3),
            "circle_fill": round(self.circle_fill, 3),
            "aspect": round(self.aspect, 2),
            "bearing_deg": round(math.degrees(self.bearing_rad), 1),
            "distance_m": None if self.distance_m is None else round(self.distance_m, 2),
            "rejected": self.rejected,
        }


def focal_length_px(width):
    return (width / 2.0) / math.tan(CAMERA_FOV_RAD / 2.0)


def colour_mask(hsv, colour):
    mask = None
    for low, high in COLOUR_RANGES[colour]:
        part = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def analyse_frame(bgr, colours=None) -> List[Detection]:
    """Return every coloured blob in the frame, accepted or rejected."""
    height, width = bgr.shape[:2]
    focal = focal_length_px(width)
    cx, horizon = width / 2.0, height / 2.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    detections = []
    for colour in colours or COLOUR_RANGES:
        mask = colour_mask(hsv, colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA_PX:
                continue
            detections.append(
                _describe(contour, area, colour, width, height, focal, cx, horizon))
    return detections


def _describe(contour, area, colour, width, height, focal, cx, horizon):
    x, y, w, h = cv2.boundingRect(contour)
    (ex, ey), radius = cv2.minEnclosingCircle(contour)
    extent = area / float(w * h)
    circle_fill = area / (math.pi * radius * radius) if radius > 0 else 0.0
    aspect = w / float(h)
    diameter = (w + h) / 2.0
    u, v = x + w / 2.0, y + h / 2.0
    det = Detection(
        object_type="unclear", colour=colour, confidence=0.0,
        image_x=u, image_y=v, diameter_px=diameter, width_px=w, height_px=h,
        area_px=area, extent=extent, circle_fill=circle_fill, aspect=aspect,
        bearing_rad=math.atan2(cx - u, focal),
    )

    if y > horizon - HORIZON_MARGIN_PX:
        det.rejected = "below_horizon"       # flat floor marking
        return det
    touches = (x <= BORDER_MARGIN_PX or y <= BORDER_MARGIN_PX
               or x + w >= width - BORDER_MARGIN_PX
               or y + h >= height - BORDER_MARGIN_PX)

    round_shape = (ROUND_EXTENT[0] <= extent <= ROUND_EXTENT[1]
                   and circle_fill >= ROUND_CIRCLE_FILL_MIN
                   and ROUND_ASPECT[0] <= aspect <= ROUND_ASPECT[1])
    upright_shape = extent >= UPRIGHT_EXTENT_MIN or circle_fill < ROUND_CIRCLE_FILL_MIN

    if round_shape:
        det.object_type = "ball"
        det.distance_m = focal * BALL_DIAMETER_M / diameter
        det.confidence = _round_confidence(extent, circle_fill, aspect, diameter)
    elif upright_shape:
        det.object_type = "upright"
    if touches:
        det.rejected = "touches_border"
    elif diameter < MIN_MEASURABLE_DIAMETER_PX:
        det.rejected = "too_small"
    return det


def _round_confidence(extent, circle_fill, aspect, diameter):
    """0-1 score of how disc-like a blob is, lower for small blobs."""
    extent_score = max(0.0, 1.0 - abs(extent - math.pi / 4) / 0.12)
    fill_score = max(0.0, min(1.0, (circle_fill - 0.6) / 0.3))
    aspect_score = max(0.0, 1.0 - abs(math.log(aspect)) / math.log(1.4))
    size_score = min(1.0, diameter / 20.0)
    return (extent_score + fill_score + aspect_score) / 3.0 * size_score


def matches(det: Detection, object_type: str, colour: Optional[str]) -> bool:
    """Does a detection match the requested target? box/cylinder -> upright."""
    if colour is not None and det.colour != colour:
        return False
    wanted = "ball" if object_type == "ball" else "upright"
    return det.object_type == wanted


def annotate(bgr, detections):
    """Copy of the frame with boxes and labels, for debug images."""
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (0, h // 2), (w - 1, h // 2), (128, 128, 128), 1)
    for det in detections:
        x = int(det.image_x - det.width_px / 2)
        y = int(det.image_y - det.height_px / 2)
        good = det.usable and det.object_type == "ball"
        colour = (0, 255, 0) if good else (0, 0, 255) if det.usable else (160, 160, 160)
        cv2.rectangle(out, (x, y), (x + det.width_px, y + det.height_px), colour, 1)
        label = f"{det.colour[0]}{det.object_type[:4]} e{det.extent:.2f} c{det.circle_fill:.2f}"
        if det.rejected:
            label += f" X{det.rejected[:6]}"
        cv2.putText(out, label, (max(0, x), max(8, y - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, colour, 1, cv2.LINE_AA)
    return out


class ObjectPerception:
    def __init__(self, camera, timestep):
        self.camera = camera
        self.camera.enable(timestep)
        self.target = None
        self.frames_examined = 0
        self.last_frame = None

    def frame(self):
        """Current camera image as a BGR array, or None."""
        raw = self.camera.getImage()
        if raw is None:
            return None
        w, h = self.camera.getWidth(), self.camera.getHeight()
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        self.last_frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return self.last_frame

    def detect(self) -> List[Detection]:
        """Analyse the current camera image."""
        image = self.frame()
        if image is None:
            return []
        self.frames_examined += 1
        return analyse_frame(image)

    def search_target(self, object_type: str, colour: str = None):
        """Best usable detection of the target in the current frame, or None."""
        candidates = [d for d in self.detect()
                      if d.usable and matches(d, object_type, colour)]
        return max(candidates, key=lambda d: d.confidence, default=None)
