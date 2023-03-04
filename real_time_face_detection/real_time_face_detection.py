# Real Time Face Detection
# Gerçek Zamanlı Yüz Algılama

import cv2

# Let's install the Haar Cascade classifier for face recognition.
# Yüz tanıma için Haar Cascade sınıflandırıcısı yükleyelim.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Let's create a video stream for the camera.
# Kamera için video akışı oluşturalım.
cap = cv2.VideoCapture(0)

while True:
    # Let's capture a frame from the camera.
    # Kameradan bir çerçeve yakalayalım.
    ret, frame = cap.read()

    # Let's convert it to gray scale image.
    # Gri ölçekli görüntüye dönüştürelim.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Let's detect faces.
    # Yüzleri tespit edelim.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Let's draw a rectangular frame around the faces.
    # Yüzlerin etrafına dörtgen çerçeve çizelim.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Let's show the image to the screen.
    # Görüntüyü ekrana gösterelim.
    cv2.imshow('frame',frame)

    # Let's exit the application by pressing the q key.
    # q tuşuna basarak uygulamadan çıkış yapalım.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Let's free the camera source and close the windows and end the process.
# Kamera kaynağını serbest bırakalım ve pencereleri kapatıp işlemi sonlandıralım.
cap.release()
cv2.destroyAllWindows()