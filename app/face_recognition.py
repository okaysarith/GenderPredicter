import numpy as np
import pandas as pd
import sklearn
import pickle
import matplotlib.pyplot as plt
import cv2 as cv

# Load all models
haar = cv.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # cascade classifier
model_svm =  pickle.load(open('./model/model_svm.pickle',mode='rb')) # machine learning model (SVM)
pca_models = pickle.load(open('./model/pca_dict.pickle',mode='rb')) # pca dictionary
model_pca = pca_models['pca'] # PCA model
mean_face_arr = pca_models['mean_face'] # Mean Face


def faceRecognitionPipeline(filename,path=True):
    if path:
        #step 1 = read image

        img = cv.imread(filename) #bgr

    else:
        img = filename#array



    # step-02: convert into gray scale
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


    # step-03: crop the face (using haar cascase classifier)

    faces = haar.detectMultiScale(gray_img,1.5,3)
    #now the classfier predicts and gives matrix
    predictions = []
    for x,y,w,h in faces:
        r=cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),(3),(1))
        roi = gray_img[y:y+h,x:x+w]
        # step-04: normalization (0-1)
        roi_n = roi/255.0

        # step-05: resize images (100,100)
        if roi_n.shape[1] > 100 :
            roi_resize = cv.resize(roi_n,(100,100),cv.INTER_AREA)
        else:
            roi_resize = cv.resize(roi_n,(100,100),cv.INTER_CUBIC)

        # step-06: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000)
        # step-07: subtract with mean
        roi_mean = roi_reshape-mean_face_arr #subtract face with  mean
        # step-08: get eigen image (apply roi_mean to pca)
        eigen_image = model_pca.transform(roi_mean)
        # step-09 Eigen Image for Visualization
        eig_img  = model_pca.inverse_transform(eigen_image)
        # step-10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        # step-11: generate report
        text = "%s : %d"%(results[0],prob_score_max*100)
        #defining color based on results
        if results[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)
        
        cv.rectangle(img,(x,y),(x+w,y+h),(color),2)
        cv.rectangle(img,(x,y-40),(x+w,y),(color),-1)
        cv.putText(img,text,(x,y),cv.FONT_HERSHEY_PLAIN,3,(255,255,255),5)
        output = {
            'roi':roi,
            'eig_img':eig_img,
            'prediction_name':results[0],
            'score':prob_score_max
        }
        predictions.append(output)

    return img,predictions

