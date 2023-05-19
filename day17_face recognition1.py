# Recognition Step

import cv2
import pandas as pd
import numpy as np
import face_recognition as fr
vid = cv2.VideoCapture(0)
fd = cv2.CascadeClassifier(
   cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

try:
    face_db = pd.read_csv('faces_data.tsv' , index_col = 0 , sep = '\t')
    data = {
        'name': face_db['name'].values.tolist(),'encoding':face_db['encoding'].values.tolist()
    }
except Exception as e:
    print(e)
    data = {'name':[] , 'encoding' : []}

while True:
    flag , img = vid.read()
    if flag:
        img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY),
        faces = fd.detectMultiScale(
            scaleFactor =  1.1 ,
            minNeighbors = 5,
            minSize = (50,50)
        )
        if len(faces) == 1:
            x,y,w,h = faces[0]
            img_face = img[y:y+h , x:x+w , :].copy()
            img_face = cv2.resize(img_face, (400,400) , interpolation = cv2.INTER_CUBIC)
            face_encoding = fr.face_encodings(img_face)
            if len(face_encoding) == 1:
                for ind,enc_value in enumerate(data['encoding']):
                    matched = fr.compare_faces(face_encoding , np.array(eval(enc_value)), tolerance =0.6)[0]
                    if matched == True:
                        cv2.putText(img , data['name'][ind] , (50,50) , cv2.FONT_HERSHEY_SIMPLEX ,1.5, (0,0,255) , 8)
                        break

        cv2.imshow('preview',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cv2.destroyAllWindows()
vid.release()