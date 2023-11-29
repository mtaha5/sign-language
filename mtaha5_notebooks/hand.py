# media pipe testing
import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break
    cv2.imshow("Image", img)
    cv2.waitkey(1)

cap.release()
cv2.destroyAllWindows()
