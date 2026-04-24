import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

# ================= CONFIG =================
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
# ==========================================

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Volume variables
previous_vol = 0

pyautogui.FAILSAFE = False

# =============== CAMERA SETUP ==============
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# ==========================================

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # Index
        x2, y2 = lmList[12][1:]  # Middle

        fingers = detector.fingersUp()

        # Draw frame box
        cv2.rectangle(img, (frameR, frameR),
                      (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # ================= MOVE =================
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

        # ================= CLICK =================
        distance = np.hypot(x2 - x1, y2 - y1)

        if fingers[1] == 1 and fingers[2] == 1 and distance < 40:
            pyautogui.click()
            time.sleep(0.3)

        # ================= VOLUME CONTROL =================
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, _ = detector.findDistance(4, 8, img)

            vol = np.interp(length, [20, 200], [0, 100])

            if abs(vol - previous_vol) > 5:  # prevent spamming
                if vol > previous_vol:
                    pyautogui.press('volumeup')
                else:
                    pyautogui.press('volumedown')

                previous_vol = vol

    # ================= FPS =================
    cTime = time.time()
    fps = int(1 / (cTime - pTime + 0.001))
    pTime = cTime

    cv2.putText(img, f'FPS: {fps}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()