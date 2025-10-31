from flask import Flask, Response
import cv2
import numpy as np
import time
import math
import sys, os, serial

# Add src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.picamera_utils import is_raspberry_camera, get_picamera

# ----------------------------------------------------------------
# Camera configuration
CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IS_RASPI_CAMERA = is_raspberry_camera()
# ----------------------------------------------------------------

# HSV threshold (initial values)
hsv_min = np.array((8, 219, 172))
hsv_max = np.array((27, 255, 255))

# Visual servoing parameters
Kp_pan  = 0.03
Kp_tilt = 0.03
theta_pan  = 90
theta_tilt = 50
theta_min, theta_max = 0, 180

fps = 0

# Serial connection to ESP8266
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    print("? Connected to ESP8266 on /dev/ttyUSB0")
except Exception as e:
    ser = None
    print(f"?? Serial connection failed: {e}")

app = Flask(__name__)

def clip(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def visualize_fps(image, fps):
    cv2.putText(image, f"FPS={fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    return image

def gen_frames():
    global fps, hsv_min, hsv_max, theta_pan, theta_tilt

    print("Using raspi camera:", IS_RASPI_CAMERA)

    if IS_RASPI_CAMERA:
        cap = get_picamera(IMAGE_WIDTH, IMAGE_HEIGHT)
        cap.start()
    else:
        cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)

    print("Camera started. Tracking ping pong ball...")

    while True:
        start_time = time.time()
        frame = cap.capture_array() if IS_RASPI_CAMERA else cap.read()[1]
        frame = cv2.blur(frame, (3, 3))

        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0]/2,
            param1=100, param2=30, minRadius=10, maxRadius=100
        )

        target_found = False

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            height, width = hsv.shape[:2]
            for (x, y, r) in circles:
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                h, s, v = hsv[y, x]
                if (hsv_min[0] <= h <= hsv_max[0] and
                    hsv_min[1] <= s <= hsv_max[1] and
                    hsv_min[2] <= v <= hsv_max[2]):

                    target_found = True
                    cx, cy = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2

                    # Draw ball center
                    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                    cv2.rectangle(output, (x-5, y-5), (x+5, y+5), (0,255,0), -1)
                    cv2.line(output, (cx, cy), (x, y), (255, 0, 0), 1)

                    # Visual servoing control
                    ex = x - cx
                    ey = y - cy
                    theta_pan  = clip(theta_pan  - Kp_pan  * ex, theta_min, theta_max)
                    theta_tilt = clip(theta_tilt + Kp_tilt * ey, theta_min, theta_max)

                    # Send to ESP8266
                    if ser:
                        ser.write(f"{int(theta_pan)},{int(theta_tilt)}\n".encode())

                    break

        if not target_found:
            cv2.putText(output, "[Warning] Tag lost...", (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # FPS calculation
        fps = 1.0 / (time.time() - start_time)
        frame = visualize_fps(output, fps)

        # Encode to JPEG and stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

    if IS_RASPI_CAMERA:
        cap.close()
    else:
        cap.release()

@app.route('/')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask stream (Visual Servoing Enabled)...")
    app.run(host='0.0.0.0', port=8000)
