"""Detector tests on synthetic 320x240 frames.

The frames are drawn with OpenCV in the world's object colours. They check
the rules (colour bands, horizon, border, shape); thresholds still need to
be confirmed on real Webots frames saved by the trace logger.
"""
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "controllers" / "project3_student"))

from perception import analyse_frame, focal_length_px, matches  # noqa: E402

W, H = 320, 240
RED = (13, 13, 242)          # BGR of baseColor 0.95 0.05 0.05
RED_BOX = (15, 15, 217)      # 0.85 0.06 0.06
RED_BOX_DARK = (10, 10, 150)  # the side face in shade
GREEN = (38, 204, 13)
ORANGE_PAD = (51, 140, 242)
MEETING_PAD = (64, 64, 242)


def background():
    img = np.zeros((H, W, 3), np.uint8)
    img[: H // 2] = (205, 200, 200)   # wall, low saturation
    img[H // 2:] = (85, 70, 60)       # carpet
    return img


def only(detections, colour):
    return [d for d in detections if d.colour == colour]


class PerceptionTests(unittest.TestCase):
    def test_red_ball_detected(self):
        img = background()
        cv2.circle(img, (160, 100), 20, RED, -1)
        dets = only(analyse_frame(img), "red")
        self.assertEqual(len(dets), 1)
        det = dets[0]
        self.assertTrue(det.usable, det.summary())
        self.assertEqual(det.object_type, "ball")
        self.assertTrue(matches(det, "ball", "red"))
        self.assertGreater(det.confidence, 0.6)
        expected = focal_length_px(W) * 0.18 / 41
        self.assertAlmostEqual(det.distance_m, expected, delta=0.15)

    def test_ball_bearing_sign(self):
        img = background()
        cv2.circle(img, (60, 100), 15, RED, -1)    # left half of the image
        det = only(analyse_frame(img), "red")[0]
        self.assertGreater(det.bearing_rad, 0)     # + = left

    def test_red_box_is_not_a_ball(self):
        img = background()
        cv2.rectangle(img, (140, 80), (180, 125), RED_BOX, -1)
        det = only(analyse_frame(img), "red")[0]
        self.assertEqual(det.object_type, "upright", det.summary())
        self.assertFalse(matches(det, "ball", "red"))

    def test_box_seen_at_an_angle_is_not_a_ball(self):
        img = background()
        cv2.rectangle(img, (130, 85), (158, 125), RED_BOX, -1)
        cv2.rectangle(img, (159, 85), (178, 125), RED_BOX_DARK, -1)
        dets = only(analyse_frame(img), "red")
        self.assertTrue(dets)
        self.assertFalse(any(d.object_type == "ball" for d in dets),
                         [d.summary() for d in dets])

    def test_small_far_box_is_not_a_ball(self):
        img = background()
        cv2.rectangle(img, (150, 108), (163, 123), RED_BOX, -1)
        dets = only(analyse_frame(img), "red")
        self.assertFalse(any(d.object_type == "ball" and d.usable for d in dets),
                         [d.summary() for d in dets])

    def test_green_ball_is_not_the_target(self):
        img = background()
        cv2.circle(img, (160, 100), 20, GREEN, -1)
        dets = analyse_frame(img)
        self.assertEqual(only(dets, "red"), [])
        green = only(dets, "green")
        self.assertEqual(green[0].object_type, "ball")
        self.assertFalse(matches(green[0], "ball", "red"))

    def test_red_floor_pad_below_horizon_rejected(self):
        img = background()
        cv2.ellipse(img, (160, 200), (120, 25), 0, 0, 360, MEETING_PAD, -1)
        dets = only(analyse_frame(img), "red")
        self.assertTrue(dets)
        self.assertTrue(all(d.rejected == "below_horizon" for d in dets))

    def test_ball_touching_red_floor_pad(self):
        # The robot standing on the red meeting-room pad: the pad fills the
        # lower image and touches the ball. Pad colour as rendered (S ~140).
        img = background()
        img[125:] = (99, 97, 216)
        cv2.circle(img, (238, 100), 25, (61, 59, 198), -1)   # ball as rendered
        balls = [d for d in only(analyse_frame(img), "red")
                 if d.usable and matches(d, "ball", "red")]
        self.assertEqual(len(balls), 1)

    def test_orange_pad_is_not_red(self):
        img = background()
        cv2.circle(img, (160, 100), 20, ORANGE_PAD, -1)
        self.assertEqual(only(analyse_frame(img), "red"), [])

    def test_ball_cut_by_image_edge_not_usable(self):
        img = background()
        cv2.circle(img, (5, 100), 20, RED, -1)
        det = only(analyse_frame(img), "red")[0]
        self.assertEqual(det.rejected, "touches_border")

    def test_tiny_blob_ignored_or_too_small(self):
        img = background()
        cv2.circle(img, (160, 112), 4, RED, -1)
        dets = only(analyse_frame(img), "red")
        self.assertTrue(all(not d.usable for d in dets))

    def test_ball_with_highlight_and_shading(self):
        img = background()
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (160, 100), 30, 255, -1)
        for row in range(70, 131):          # darker towards the bottom
            shade = int(242 - (row - 70) * 1.8)
            img[row][mask[row] > 0] = (13, 13, shade)
        cv2.circle(img, (150, 88), 5, (230, 230, 255), -1)   # specular spot
        det = only(analyse_frame(img), "red")[0]
        self.assertEqual(det.object_type, "ball", det.summary())
        self.assertTrue(det.usable)

    def test_ball_and_box_side_by_side(self):
        img = background()
        cv2.circle(img, (90, 100), 18, RED, -1)
        cv2.rectangle(img, (200, 85), (232, 122), RED_BOX, -1)
        dets = only(analyse_frame(img), "red")
        balls = [d for d in dets if matches(d, "ball", "red") and d.usable]
        self.assertEqual(len(balls), 1)
        self.assertLess(balls[0].image_x, 160)

    def test_mid_distance_ball(self):
        img = background()
        r = 9                                 # about 3.6 m away
        cv2.circle(img, (200, 114), r, RED, -1)
        det = only(analyse_frame(img), "red")[0]
        self.assertEqual(det.object_type, "ball", det.summary())
        self.assertTrue(det.usable)
        self.assertGreater(det.distance_m, 2.5)


if __name__ == "__main__":
    unittest.main()
