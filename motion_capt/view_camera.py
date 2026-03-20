import pyrealsense2 as rs
import numpy as np
import cv2
import time

# mediapipe 0.10.x explicit impo

from mediapipe.python.solutions import hands as mp_hands_mod
from mediapipe.python.solutions import drawing_utils as mp_draw_mod
# ── RealSense: depth + infrared (both on stereo module, no CMIO conflict) ─────
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth,    640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8,  30)
pipeline.start(config)

scale     = (pipeline.get_active_profile()
                     .get_device()
                     .first_depth_sensor()
                     .get_depth_scale())
colorizer = rs.colorizer()
prev_time = time.time()


def depth_at_landmark(lm, depth_img):
    dh, dw = depth_img.shape
    cx = int(np.clip(lm.x * dw, 0, dw - 1))
    cy = int(np.clip(lm.y * dh, 0, dh - 1))
    x0, x1 = max(0, cx - 3), min(dw, cx + 4)
    y0, y1 = max(0, cy - 3), min(dh, cy + 4)
    patch = depth_img[y0:y1, x0:x1].flatten()
    patch = patch[patch > 0]
    return float(np.median(patch)) * scale if patch.size > 0 else 0.0


def pointing_direction(lm):
    middle_curled = lm.landmark[12].y > lm.landmark[10].y
    ring_curled   = lm.landmark[16].y > lm.landmark[14].y
    pinky_curled  = lm.landmark[20].y > lm.landmark[18].y
    if not (middle_curled and ring_curled and pinky_curled):
        return None
    dx = lm.landmark[8].x - lm.landmark[5].x
    dy = lm.landmark[8].y - lm.landmark[5].y
    if abs(dx) > abs(dy):
        return "LEFT" if dx < -0.08 else "RIGHT"
    else:
        return "UP"   if dy < -0.08 else "DOWN"


try:
    with mp_hands_mod.Hands(max_num_hands=2,
                             min_detection_confidence=0.7,
                             min_tracking_confidence=0.5) as hands:
        while True:
            frames = pipeline.wait_for_frames(10000)
            depth_frame = frames.get_depth_frame()
            ir_frame    = frames.get_infrared_frame()
            if not depth_frame or not ir_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())   # 640×480 uint16
            ir_img    = np.asanyarray(ir_frame.get_data())       # 640×480 uint8

            # Convert IR grayscale → 3-channel for MediaPipe
            ir_rgb = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
            result = hands.process(ir_rgb)

            # Colorized depth for display
            depth_vis = cv2.cvtColor(
                np.asanyarray(colorizer.colorize(depth_frame).get_data()),
                cv2.COLOR_RGB2BGR)

            # IR display (brighter)
            ir_display = cv2.cvtColor(
                cv2.equalizeHist(ir_img), cv2.COLOR_GRAY2BGR)

            gesture_text  = ""
            gesture_color = (255, 255, 255)

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, handedness in zip(result.multi_hand_landmarks,
                                          result.multi_handedness):
                    side      = handedness.classification[0].label
                    dist      = depth_at_landmark(lm.landmark[0], depth_img)
                    direction = pointing_direction(lm)

                    mp_draw_mod.draw_landmarks(
                        ir_display, lm,
                        mp_hands_mod.HAND_CONNECTIONS)

                    wx = int(lm.landmark[0].x * ir_display.shape[1])
                    wy = int(lm.landmark[0].y * ir_display.shape[0])
                    cv2.putText(ir_display, f"{side} {dist:.2f}m",
                                (wx + 10, wy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 200, 255), 2)

                    if direction:
                        if side == "Right" and direction == "LEFT":
                            gesture_text  = "RIGHT HAND -> LEFT"
                            gesture_color = (0, 255, 0)
                        else:
                            gesture_text  = f"{side} -> {direction}"
                            gesture_color = (0, 200, 255)

            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            for img in [ir_display, depth_vis]:
                cv2.putText(img, f"{fps:.1f} fps", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if gesture_text:
                cv2.putText(ir_display, gesture_text,
                            (10, ir_display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, gesture_color, 3)

            cv2.imshow("IR + Gesture | Depth",
                       np.hstack([ir_display, depth_vis]))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()