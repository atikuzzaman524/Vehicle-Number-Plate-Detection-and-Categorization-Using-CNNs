# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 06:05:50 2018

@author: atikuzzaman
"""


import time
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import model_from_json
from keras.models import load_model
font = cv2.FONT_HERSHEY_SIMPLEX



img_rows=25
img_cols=25
num_channel=1
model = load_model('model.hdf5')


plate_cascade = cv2.CascadeClassifier("CarPlate.xml")
#eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


#load Video from PC
cap = cv2.VideoCapture("E:/ThesisBookFinal/FinalCode/ExtraFinal/trrim/MVI_7018-4.mov")

#for capture real videos on camera
#cap = cv2.VideoCapture(0)
#detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(1,1))

while True :
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    for (x,y,w,h) in faces:
        
        crop_img = img[y-5:y+h+10, x-5:x+w+10]
        cv2.rectangle(img,(x-5,y-5),(x+w+10,y+h+10),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imshow('CropImg',crop_img)
    
            
        plate_image=cv2.resize(crop_img,(180,100))
            #classCrop
        test_image = plate_image[20:55, 140:175]
        cv2.imshow('CLass',test_image)
            
        test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image=cv2.resize(test_image,(img_rows,img_cols))
        test_image = np.array(test_image)
        test_image = test_image.astype('float32')
        test_image /= 255
            #print (test_image.shape)
            
        if num_channel==1:
            if K.image_dim_ordering()=='th':
            	test_image= np.expand_dims(test_image, axis=0)
            	test_image= np.expand_dims(test_image, axis=0)
            		#print (test_image.shape)
            else:
            	test_image= np.expand_dims(test_image, axis=3) 
            	test_image= np.expand_dims(test_image, axis=0)
            		#print (test_image.shape)
            		
        else:
            if K.image_dim_ordering()=='th':
            	test_image=np.rollaxis(test_image,2,0)
            	test_image= np.expand_dims(test_image, axis=0)
            		#print (test_image.shape)
            else:
            	test_image= np.expand_dims(test_image, axis=0)
            		#print (test_image.shape)
            		
            # Predicting the test image
        print((model.predict(test_image)))
        print(model.predict_classes(test_image))
        x=model.predict_classes(test_image)
        #y= str(x[0])
        if x[0]==0:
            y="CHA"
        elif x[0]==1:
            y="GA"
        elif x[0]==2:
            y = "GHA"
        elif x[0]==3:
            y = "KHA"
        cv2.putText(img,y,(200,200), font, 4,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()