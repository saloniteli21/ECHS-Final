import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=85, trackCon=50):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = int(detectionCon)  # Pass as integers between 0 and 100
        self.trackCon = int(trackCon)  # Pass as integers between 0 and 100

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)  # Pass as integers
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the BGR image to RGB for mediapipe processing
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If hands are detected, draw landmarks
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True, color=(255, 0, 255), z_axis=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                if not z_axis:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                else:
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z, 3)
                    lmList.append([id, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Change to 0 for the default camera
    detector = handDetector(maxHands=1)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img, z_axis=True, draw=False)
        
        if len(lmList) != 0:
            print(lmList[4])  # Printing the position of the thumb tip (landmark 4)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Show the image with landmarks
        cv2.imshow("Image", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

