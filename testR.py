
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    raise SystemExit("Kamera tidak terdeteksi")

profiles = set()

for sensor in devices[0].query_sensors():
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() == rs.stream.color:
            video = profile.as_video_stream_profile()
            profiles.add((
                video.width(),
                video.height(),
                profile.fps(),
                str(profile.format())
            ))

for width, height, fps, fmt in sorted(profiles):
    print(f"{width} x {height} | {fps} FPS | {fmt}")
