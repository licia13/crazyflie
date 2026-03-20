import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.devices
if len(devices) == 0:
    print("No RealSense devices found!")
else:
    for d in devices:
        print(f"Device: {d.get_info(rs.camera_info.name)}")
        print(f"Serial: {d.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {d.get_info(rs.camera_info.firmware_version)}")
        print()
        for s in d.sensors:
            print(f"  Sensor: {s.get_info(rs.camera_info.name)}")
            for p in s.get_stream_profiles():
                vp = p.as_video_stream_profile()
                if p.stream_type() == rs.stream.depth:
                    print(f"    DEPTH  {vp.width():4}x{vp.height():3} @ {p.fps():2}fps  {p.format()}")
                elif p.stream_type() == rs.stream.color:
                    print(f"    COLOR  {vp.width():4}x{vp.height():3} @ {p.fps():2}fps  {p.format()}")
            print()
