import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 6)
pipeline.start(config)

# Store last mouse position
mouse_x, mouse_y = 128, 72

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

cv2.namedWindow('Depth Test')
cv2.setMouseCallback('Depth Test', on_mouse)

print("Move mouse over window to measure distance at any pixel.")
print("Center pixel distance will print to terminal every second.")

try:
    while True:
        frames = pipeline.wait_for_frames(10000)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # --- Distance measurements ---
        center_dist = depth_frame.get_distance(128, 72)
        mouse_dist  = depth_frame.get_distance(mouse_x, mouse_y)

        print(f"Center: {center_dist:.3f} m | Mouse ({mouse_x},{mouse_y}): {mouse_dist:.3f} m")

        # --- Visualisation ---
        depth_image = np.asanyarray(depth_frame.get_data())
        display = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)

        # Scale display up so it's easier to see (256x144 is tiny)
        display = cv2.resize(display, (768, 432))
        scale = 3  # because we scaled 3x

        # Draw crosshair at center
        cx, cy = 128 * scale, 72 * scale
        cv2.drawMarker(display, (cx, cy), (255,255,255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(display, f"{center_dist:.3f} m", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Draw dot at mouse position
        mx, my = mouse_x * scale, mouse_y * scale
        cv2.circle(display, (mx, my), 5, (0, 255, 255), -1)
        cv2.putText(display, f"{mouse_dist:.3f} m", (mx + 10, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Depth Test', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()