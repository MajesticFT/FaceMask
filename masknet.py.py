import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt

mask_label = {0:'MASK',1:'NO MASK'}
model = keras.models.load_model('/content/masknet.h5')

face_model = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    img = cv2.imread(frame)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
    cv2.imshow("Face Location Draw rectangle",frame)
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
        cv2.rectangle(new_img,(x,y),(x+w,y+h),dist_label[label[i]],1)
    plt.imshow(new_img)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()