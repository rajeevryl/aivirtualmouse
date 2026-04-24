import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui
import math

print("VIRTUAL TOUCH STARTED")

# ================= CONFIG =================
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
# ==========================================

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

dragging = False
lastActionTime = 0
prevZoomDist = 0

pyautogui.FAILSAFE = False

# =============== CAMERA SETUP ==============
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# ==========================================

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

cv2.namedWindow("AI Virtual Mouse", cv2.WINDOW_NORMAL)

# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    statusText = ""

    if lmList and len(lmList) >= 21:
        x1, y1 = lmList[8][1:]   # Index
        x2, y2 = lmList[12][1:]  # Middle
        x3, y3 = lmList[16][1:]  # Ring
        xThumb, yThumb = lmList[4][1:]

        fingers = detector.fingersUp()

        # Draw frame
        cv2.rectangle(img, (frameR, frameR),
                      (wCam-frameR, hCam-frameR),
                      (255, 0, 255), 2)

        # ================= MOVE =================
        if fingers == [0,1,0,0,0]:
            xScr = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            yScr = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            clocX = plocX + (xScr - plocX) / smoothening
            clocY = plocY + (yScr - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)

            plocX, plocY = clocX, clocY
            statusText = "Move"

        # ================= LEFT CLICK =================
        dist_im = math.hypot(x1 - x2, y1 - y2)
        if fingers == [0,1,1,0,0] and dist_im < 30 and time.time() - lastActionTime > 0.4:
            pyautogui.click()
            lastActionTime = time.time()
            statusText = "Left Click"

        # ================= RIGHT CLICK =================
        dist_ir = math.hypot(x1 - x3, y1 - y3)
        if fingers == [0,1,1,1,0] and dist_im < 30 and dist_ir < 40 and time.time() - lastActionTime > 0.5:
            pyautogui.click(button='right')
            lastActionTime = time.time()
            statusText = "Right Click"

        # ================= DRAG =================
        pinch = math.hypot(x1 - xThumb, y1 - yThumb)
        if pinch < 30 and fingers[1] == 1:
            if not dragging:
                dragging = True
                pyautogui.mouseDown()

            xScr = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            yScr = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            clocX = plocX + (xScr - plocX) / smoothening
            clocY = plocY + (yScr - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY
            statusText = "Dragging"

        else:
            if dragging:
                dragging = False
                pyautogui.mouseUp()
                statusText = "Drop"

        # ================= SCROLL =================
        if fingers == [1,0,0,0,0] and time.time() - lastActionTime > 0.2:
            if yThumb < y1 - 20:
                pyautogui.scroll(40)
                statusText = "Scroll Up"
            elif yThumb > y1 + 20:
                pyautogui.scroll(-40)
                statusText = "Scroll Down"
            lastActionTime = time.time()

        # ================= ZOOM =================
        if fingers[:3] == [1,1,1] and time.time() - lastActionTime > 0.3:
            zoomDist = math.hypot(x1 - xThumb, y1 - yThumb)

            if prevZoomDist != 0:
                if zoomDist < prevZoomDist - 5:
                    pyautogui.hotkey("ctrl", "+")
                    statusText = "Zoom In"
                elif zoomDist > prevZoomDist + 5:
                    pyautogui.hotkey("ctrl", "-")
                    statusText = "Zoom Out"

                lastActionTime = time.time()

            prevZoomDist = zoomDist

    # ================= FPS =================
    cTime = time.time()
    fps = int(1 / (cTime - pTime + 0.001))
    pTime = cTime

    cv2.putText(img, f"FPS: {fps}", (20,40),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.putText(img, f"Mode: {statusText}", (20,80),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()