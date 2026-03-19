import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
pipeline.start(config)

print("Depth stream running. Press Q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames(10000)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        dist = depth_frame.get_distance(128, 72)
        print(f"Center distance: {dist:.3f} m")

        depth_image = np.asanyarray(depth_frame.get_data())
        display = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
        display = cv2.resize(display, (768, 432))
        cv2.imshow('Depth', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()