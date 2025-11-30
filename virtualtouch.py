import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import os
import pyautogui

#############################
wCam, hCam = 640, 480  # You can reduce to 320,240 if CPU is slow
frameR = 100  # Frame Reduction
smoothening = 7
##############################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

cv2.namedWindow("AI Virtual Mouse", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        continue  # Skip frame if webcam fails

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Move mode: only index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            if abs(clocX - plocX) > 2 or abs(clocY - plocY) > 2:
                try:
                    autopy.mouse.move(wScr - clocX, clocY)
                except Exception as e:
                    print("Autopy move error:", e)
            plocX, plocY = clocX, clocY
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # Left click: index + middle up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            distance = np.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
            if distance < 40:
                try:
                    autopy.mouse.click()
                except Exception as e:
                    print("Autopy click error:", e)

        # Right click: index+middle+ring up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
            distance = np.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
            if distance < 40:
                try:
                    autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
                except Exception as e:
                    print("Autopy right click error:", e)

        # Open A drive: index + pinky up
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            distance = np.hypot(lmList[8][1] - lmList[20][1], lmList[8][2] - lmList[20][2])
            if distance < 40:
                try:
                    os.startfile("A:\\")
                except Exception as e:
                    print("Error opening drive:", e)
               
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("AI Virtual Mouse", img)

    # Small sleep to reduce CPU usage
    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
