print("1. Program mulai", flush=True)

import cv2
import numpy as np
import pyrealsense2 as rs

print("2. Import berhasil", flush=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, rs.format.bgr8)

started = False

try:
    print("3. Membuka kamera...", flush=True)
    pipeline.start(config)
    started = True
    print("4. Kamera aktif, menunggu frame...", flush=True)

    first_frame = True

    while True:
        frames = pipeline.wait_for_frames(10000)
        color = frames.get_color_frame()

        if not color:
            continue

        if first_frame:
            print("5. Frame diterima, membuka jendela...", flush=True)

        image = np.asanyarray(color.get_data())
        cv2.imshow("RealSense D415 - RGB", image)
        key = cv2.waitKey(1) & 0xFF

        if first_frame:
            print("6. Tampilan diproses. Tekan Q untuk keluar.", flush=True)
            first_frame = False

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("Dihentikan pengguna.", flush=True)
except Exception as error:
    print(f"ERROR: {type(error).__name__}: {error}", flush=True)
finally:
    if started:
        pipeline.stop()
    cv2.destroyAllWindows()