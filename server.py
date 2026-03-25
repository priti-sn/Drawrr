import threading
import time
import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class Camera:
    """Threaded camera reader so we don't block on VideoCapture.read()."""

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.grabbed = ok
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def release(self):
        self.stopped = True
        self.cap.release()


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_up(landmarks, tip, pip):
    # lower y means higher on screen
    return landmarks[tip].y < landmarks[pip].y


def detect_gesture(lms):
    thumb_tip = lms[4]
    idx_tip = lms[8]

    # pinch check
    if _dist(thumb_tip, idx_tip) < 0.045:
        return "pinch"

    # fist = all four fingers curled
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    all_down = all(not _finger_up(lms, t, p) for t, p in pairs)
    if all_down:
        return "fist"

    # pointing = index extended, rest curled
    if (_finger_up(lms, 8, 6)
            and not _finger_up(lms, 12, 10)
            and not _finger_up(lms, 16, 14)
            and not _finger_up(lms, 20, 18)):
        return "draw"

    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_feed(ws: WebSocket):
    await ws.accept()

    cam = Camera(0)

    # BGRA canvas so we can alpha-blend drawings onto the camera frame
    canvas = np.zeros((480, 640, 4), dtype=np.uint8)
    pen_on = True
    prev_pt = None

    pinch_held = False  # track rising edge of pinch

    fist_t0 = None
    FIST_HOLD_SEC = 0.8

    # mediapipe hand landmarker
    base_opts = python.BaseOptions(model_asset_path='hand_landmarker.task')
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_detector = vision.HandLandmarker.create_from_options(opts)

    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, 75]

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                await ws.send_bytes(b"")
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand_detector.detect(mp_img)

            gesture = None
            idx_pos = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                gesture = detect_gesture(hand)

                if gesture in ("draw", None):
                    lm = hand[8]
                    idx_pos = (int(lm.x * w), int(lm.y * h))

            now = time.time()

            # pinch toggles pen (only on the initial pinch, not while held)
            if gesture == "pinch":
                if not pinch_held:
                    pen_on = not pen_on
                    pinch_held = True
                    prev_pt = None
            else:
                pinch_held = False

            # fist held long enough => clear
            if gesture == "fist":
                if fist_t0 is None:
                    fist_t0 = now
                elif now - fist_t0 > FIST_HOLD_SEC:
                    canvas[:] = 0
                    prev_pt = None
                    fist_t0 = None
            else:
                fist_t0 = None

            # draw line segments
            if pen_on and gesture == "draw" and idx_pos:
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, idx_pos,
                             (0, 200, 255, 255), 3, cv2.LINE_AA)
                prev_pt = idx_pos
            else:
                prev_pt = None

            # overlay canvas onto camera frame
            alpha = canvas[:, :, 3] / 255.0
            for c in range(3):
                frame[:, :, c] = (
                    alpha * canvas[:, :, c] + (1 - alpha) * frame[:, :, c]
                ).astype(np.uint8)

            # cursor dot
            if idx_pos:
                clr = (0, 200, 255) if pen_on else (128, 128, 128)
                cv2.circle(frame, idx_pos, 6, clr, -1, cv2.LINE_AA)

            # status label
            label = "pen: on" if pen_on else "pen: off"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2,
                        cv2.LINE_AA)

            # clear progress bar
            if gesture == "fist" and fist_t0 is not None:
                elapsed = now - fist_t0
                pct = min(elapsed / FIST_HOLD_SEC, 1.0)
                bar_w = int(200 * pct)
                cv2.rectangle(frame, (10, 45), (10 + bar_w, 60),
                              (0, 0, 255), -1)
                cv2.rectangle(frame, (10, 45), (210, 60),
                              (255, 255, 255), 1)
                cv2.putText(frame, "CLEARING...", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (155, 155, 155), 1, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", frame, jpg_params)
            await ws.send_bytes(buf.tobytes())

    except WebSocketDisconnect:
        pass
    finally:
        hand_detector.close()
        cam.release()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
