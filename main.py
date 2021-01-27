import cv2 
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 50

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
model = tf.keras.models.load_model('maskDetector.model')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))

while True:

    ret, image = capture.read()
    if image is not None:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImage, 1.3, 5)


        for (x, y, w, h) in faces:

            if faces is not ():

                regionOfImage = grayImage[y:y+h, x:x+w]
                regionOfImage = cv2.resize(regionOfImage, (IMAGE_SIZE, IMAGE_SIZE))
                regionOfImage = np.array(regionOfImage).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
                regionOfImage = regionOfImage / 255
                
                prediction = model.predict([regionOfImage])
                print(prediction[0])

                if prediction[0] > 0.025:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
                    cv2.rectangle(image, (x-2, y), (x+w+2, y-60), (0, 255, 0), -1)
                    cv2.putText(image, 'Mask Detected', (x+10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.66, (0,0,0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 4)
                    cv2.rectangle(image, (x-2, y), (x+w+2, y-60), (0, 0, 255), -1)
                    cv2.putText(image, 'No Mask Detected', (x+10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.66, (0,0,0), 2, cv2.LINE_AA)
                
        output.write(image)

        cv2.imshow('image', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break 

capture.release()
output.release()
cv2.destroyAllWindows()