import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 648)  # ubah lebar cam
cam.set(3, 488)  # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)
    cv2.imshow('Webcamku', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()



