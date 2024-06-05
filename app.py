from flask import Flask, render_template, Response
import cv2
import time
import numpy as np
import math
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import handtrackmodule as htm

app = Flask(__name__)
detector = htm.handDetector(detectionCon=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

def gen_frames():
    print("call")
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 1080)
    pTime = 0

    vol = 0
    volBar = 400
    volPer = 0

    while True:
        print("run")
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 6)

        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            print("Failed to encode frame.")
            continue

        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    cap.release()
    print("Camera released.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("run11")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,port=9080)
