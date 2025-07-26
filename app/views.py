from flask import render_template,request
import os
import cv2 as cv
from app.face_recognition import faceRecognitionPipeline
import matplotlib.image as matimg

UPLOAD_FOLDER = 'static/upload'





def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method == 'POST':
        f= request.files['image_name']#this is the key which we use to call the file
        filename = f.filename

        #save our image in upload folder

        path = os.path.join(UPLOAD_FOLDER,filename)#joins folder and filename 
        f.save(path) #save iamge into upload folder 

        #get predictions
        pred_image,predictions = faceRecognitionPipeline(path)
        pred_filename = 'predicted file.jpg'
        cv.imwrite(f'./static/predict/{pred_filename}',pred_image)

        # for generating report 
        report = []
        for i , obj in enumerate(predictions):
            gray_img = obj['roi'] #grayscale image (array)
            eigen_img = obj['eig_img'].reshape(100,100)#eigen img array
            gender_name= obj['prediction_name']#name

            score = round(obj['score']*100,2)

            #save grayscale and eigen in predictfolder

            gray_image_name= f'roi_{i}.jpg'
            eigen_image_name= f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_img,cmap='gray')
            matimg.imsave(f'./static/predict/{eigen_image_name}',eigen_img,cmap='gray')

            #saveing report

            report.append([gray_image_name,
                           eigen_image_name,
                           gender_name,score])

        return render_template('gender.html',fileupload=True,report=report)#POST request  
     

    return render_template('gender.html')#GET request     
            
            
            

print('ML model predicted successfully')
      
        

    